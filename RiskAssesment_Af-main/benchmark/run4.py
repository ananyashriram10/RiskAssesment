"""Round 4: every variant is timing-free; adds numeric embeddings, multi-aggregator message passing
with jumping knowledge, and a pairwise ranking loss, on the official SMEsD split.

  python3 -m benchmark.run4 --variant T0 --tag r4_T0 --seeds 5
  python3 -m benchmark.run4 --variant TEAR --tag r4_TEAR --seeds 10

Variants are declared below, before any is run. Epochs are chosen on validation AUC and the headline is
chosen on validation (benchmark.report4). Test predictions are saved but the per-seed log prints
validation only. Predictions: results/preds_<tag>.csv (+ __valid); summary: results/info_<tag>.json.

Inputs: company attributes and lawsuit cause/court/result counts. No lawsuit timing columns
(log recent lawsuits, newest age, mean age), no hand-made degree features, no ComRisk embeddings.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from benchmark import data as D
from benchmark.model_ext import SMEModel
from benchmark.run2 import BASE, LAWSUIT_DIM, P_CFG, build, score
from risk_gnn import GateConfig, quantile_edges

OUT = Path(__file__).resolve().parents[1] / "results"
TIMING = [23, 24, 25]                      # log recent lawsuits, newest lawsuit age, mean lawsuit age
AGE = 2                                    # set-up age (outcome-aligned observation date)
E_CFG = dict(numeric_embedding="ple", embedding_bins=8, embedding_dim=4)
A_CFG = dict(message_passing_layers=2, multi_aggregate=True, degree_mlp=True, jumping_knowledge=True)
TWO_LAYERS = dict(message_passing_layers=2)
VARIANTS = {
    "T0": dict(drop=TIMING),                                                   # timing-free reference
    "T2": dict(drop=TIMING, cfg=TWO_LAYERS),                                   # control for the layer count in TA
    "TP": dict(drop=TIMING, cfg=P_CFG),                                        # round-3 sector blend, timing-free
    "TL": dict(drop=TIMING, lawsuit=True, lawsuit_timing=False),               # lawsuit vectors without age inputs
    "TE": dict(drop=TIMING, cfg=E_CFG),                                        # numeric embedding
    "TA": dict(drop=TIMING, cfg=A_CFG),                                        # multi-aggregator + degree MLP + JK
    "TR": dict(drop=TIMING, rank=1.0),                                         # ranking loss
    "TEAR": dict(drop=TIMING, cfg={**E_CFG, **A_CFG}, rank=1.0),               # E + A + R
    "TLEAR": dict(drop=TIMING, lawsuit=True, lawsuit_timing=False, cfg={**E_CFG, **A_CFG}, rank=1.0),
    "T0NA": dict(drop=TIMING + [AGE]),                                         # diagnostic: also drop set-up age
}


def pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean logistic loss over (bankrupt, surviving) pairs: a smooth surrogate for 1 - AUC."""
    positive, negative = logits[target > 0.5], logits[target <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    return F.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean()


def fit_one(splits, variant: dict, seed: int, args):
    snaps, vocab, seq = build(splits, variant)
    skip = vocab["__person__"]
    y = {k: torch.tensor(D.labels(s), dtype=torch.float32) for k, s in splits.items()}
    n_comp = {k: len(s["ids"]) for k, s in splits.items()}
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = dict(BASE); cfg.update(variant.get("cfg", {})); cfg["dropout"] = args.dropout
    config = GateConfig(**cfg)
    edges = None
    if config.numeric_embedding == "ple":          # bin edges from training companies only
        edges = quantile_edges(snaps["train"].features[:n_comp["train"]], config.embedding_bins)
    model = SMEModel(snaps["train"].features.shape[1], len(vocab), len(D.RELATIONS), config,
                     LAWSUIT_DIM if variant.get("lawsuit") else None, edges, variant.get("lawsuit_timing", True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    sectors = [s for s in torch.unique(snaps["train"].sectors).tolist() if s != skip]
    rng = np.random.default_rng(seed)
    rank_weight = variant.get("rank", 0.0)
    best_auc, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(sectors)
        for start in range(0, len(order), args.batch_sectors):
            optimizer.zero_grad(set_to_none=True)
            logits, targets, aux = [], [], 0.0
            for sector in order[start:start + args.batch_sectors]:
                o = model(snaps["train"], int(sector), seq["train"] if seq else None)
                logits.append(o.logits); targets.append(y["train"][o.target_indices])
                if model.gnn.aux_loss is not None:
                    aux = aux + model.gnn.aux_loss * len(o.logits)
            logits, target = torch.cat(logits), torch.cat(targets)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") + aux
            if rank_weight:                      # pairs span every sector in the batch, as the AUC does
                loss = loss + rank_weight * len(target) * pairwise_rank_loss(logits, target)
            (loss / max(len(target), 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        valid_scores = score(model, snaps["valid"], seq["valid"] if seq else None, n_comp["valid"], skip)
        auc = roc_auc_score(D.labels(splits["valid"]), valid_scores)
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    valid = score(model, snaps["valid"], seq["valid"] if seq else None, n_comp["valid"], skip)
    test, gates = score(model, snaps["test"], seq["test"] if seq else None, n_comp["test"], skip, collect_gates=True)
    return {"valid": valid, "test": test, "valid_auc": best_auc, "epoch": best_epoch, "gates": gates}


def main(args) -> None:
    torch.set_num_threads(args.threads)
    variant = VARIANTS[args.variant]
    splits = D.raw_splits()
    OUT.mkdir(exist_ok=True)
    started, runs = time.monotonic(), []
    for seed in range(args.seeds):
        r = fit_one(splits, variant, 42 + seed, args)
        runs.append(r)
        print(f"{args.tag} seed {42 + seed}: epoch {r['epoch']} valid AUC {r['valid_auc']:.4f} ({time.monotonic() - started:.0f}s)", flush=True)
    test = np.mean([r["test"] for r in runs], axis=0); valid = np.mean([r["valid"] for r in runs], axis=0)
    for name, split, scores in (("", splits["test"], test), ("__valid", splits["valid"], valid)):
        pd.DataFrame({"company_id": split["ids"], "label": split["labels"], "score": scores}).to_csv(
            OUT / f"preds_{args.tag}{name}.csv", index=False)
    gates = [r["gates"] for r in runs if r["gates"] is not None]
    info = {"tag": args.tag, "variant": args.variant, "dropout": args.dropout, "wd": args.wd, "seeds": args.seeds,
            "epochs": [r["epoch"] for r in runs], "seed_valid_auc": [r["valid_auc"] for r in runs],
            "seed_test_auc": [roc_auc_score(D.labels(splits["test"]), r["test"]) for r in runs],
            "ensemble_valid_auc": roc_auc_score(D.labels(splits["valid"]), valid),
            "ensemble_test_auc": roc_auc_score(D.labels(splits["test"]), test),
            "gates_mean_over_seeds": ({"in": np.mean([g["in"] for g in gates], axis=0).tolist(),
                                       "out": np.mean([g["out"] for g in gates], axis=0).tolist()} if gates else None),
            "relations": list(D.RELATIONS)}
    (OUT / f"info_{args.tag}.json").write_text(json.dumps(info, indent=2))
    print(f"{args.tag}: ensemble valid AUC {info['ensemble_valid_auc']:.4f} (test saved; read it from benchmark.report4)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-sectors", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    main(parser.parse_args())

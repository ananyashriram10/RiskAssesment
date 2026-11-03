"""Round 3: regularized training plus lawsuit vectors, a hypergraph layer, and the
sector-conditioned person blend, on the official SMEsD split.

  python3 -m benchmark.run2 --variant base --tag r_d1_w1e4 --dropout 0.1 --wd 1e-4 --seeds 5
  python3 -m benchmark.run2 --variant LHP --tag final_LHP --dropout 0.3 --wd 1e-3 --seeds 10

Everything is chosen on the validation split; the test split is scored once per final model.
Predictions: results/preds_<tag>.csv (+ __valid); summary: results/info_<tag>.json.
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
from sklearn.preprocessing import RobustScaler
from torch.nn import functional as F

from benchmark import data as D
from benchmark.model_ext import SMEModel
from risk_gnn import GateConfig

OUT = Path(__file__).resolve().parents[1] / "results"
BASE = dict(hidden_dim=32, max_context_nodes=32, max_learned_neighbors=2, max_incoming_edges=64,
            message_passing_layers=1, aggregation="mean", symmetric_documented=True, true_mean=True,
            edge_mode="documented", candidate_mode="encoder", adjacent_context=True, degree_aware=True)
P_CFG = dict(message_passing_layers=2, two_hop_context=True, sector_blend=True, blend_l1=0.1)
VARIANTS = {
    "base": dict(),
    "L": dict(lawsuit=True),
    "H": dict(cfg=dict(hyper_layer=True)),
    "P0": dict(cfg=dict(message_passing_layers=2, two_hop_context=True)),
    "P": dict(cfg=P_CFG),
    "LH": dict(lawsuit=True, cfg=dict(hyper_layer=True)),
    "LHP": dict(lawsuit=True, cfg=dict(hyper_layer=True, **P_CFG)),
    # Control: same model as P but with no L1 penalty, to see whether the data wants any out-of-sector weight.
    "Pl0": dict(cfg=dict(P_CFG, blend_l1=0.0)),
    # Diagnostics: drop the three lawsuit timing columns (log recent lawsuits, newest age, mean age).
    "NT": dict(drop=[23, 24, 25]),
    # Follow-up candidate: exclude outcome-aligned timing and expose relation counts to the GNN.
    "PNTD": dict(drop=[23, 24, 25], degree=True, cfg=P_CFG),
    "PNTDG": dict(drop=[23, 24, 25], degree=True, extras=("meta", "group"), cfg=P_CFG),
}
LAWSUIT_DIM = 16


def build(splits, variant: dict):
    X = {k: D.node_features(s) for k, s in splits.items()}
    if variant.get("drop"):                          # diagnostic: remove columns (e.g. lawsuit timing features)
        keep = [i for i in range(X["train"].shape[1]) if i not in variant["drop"]]
        X = {k: v[:, keep] for k, v in X.items()}
    if variant.get("degree"):
        X = {k: np.column_stack([X[k], D.degree_features(splits[k])]) for k in splits}
    if "meta" in variant.get("extras", ()):
        X = {k: np.column_stack([X[k], D.meta_features(splits[k])]) for k in splits}
    if "group" in variant.get("extras", ()):
        X = {k: np.column_stack([X[k], D.group_features(splits[k], D.node_features(splits[k]),
                                                 D.degree_features(splits[k])[:, -1])]) for k in splits}
    if variant.get("lawsuit"):                       # lawsuit information comes from the vector encoder instead
        X = {k: v[:, :3] for k, v in X.items()}
    scaler = RobustScaler().fit(X["train"])
    Z = {k: np.clip(scaler.transform(v), -5, 5) for k, v in X.items()}
    vocab: dict[str, int] = {}
    for s in splits.values():
        for g in D.industries(s):
            vocab.setdefault(g, len(vocab))
    vocab.setdefault("__person__", len(vocab))
    snaps = {k: D.make_hetero_snapshot(s, Z[k], vocab) for k, s in splits.items()}
    seq = {k: D.lawsuit_tensors(s) for k, s in splits.items()} if variant.get("lawsuit") else None
    return snaps, vocab, seq


def score(model, snap, seq, n_companies: int, skip: int, collect_gates=False):
    model.eval()
    out = np.full(len(snap.company_ids), np.nan)
    gates, sizes = [], []
    with torch.no_grad():
        for sector in torch.unique(snap.sectors).tolist():
            if sector == skip:
                continue
            o = model(snap, sector, seq)
            out[o.target_indices.numpy()] = o.logits.numpy()
            if collect_gates and model.gnn.last_gates is not None:
                gates.append(torch.stack(model.gnn.last_gates).numpy()); sizes.append(len(o.target_indices))
    scores = out[:n_companies]
    if collect_gates:
        if not gates:
            return scores, None
        w = np.asarray(sizes, dtype=float) / sum(sizes)
        arr = np.stack(gates)                                  # sectors x 2 x relations
        return scores, {"in": (arr[:, 0] * w[:, None]).sum(0).tolist(), "out": (arr[:, 1] * w[:, None]).sum(0).tolist(),
                        "out_max_over_sectors": arr[:, 1].max(0).tolist()}
    return scores


def fit_one(splits, variant: dict, seed: int, args):
    snaps, vocab, seq = build(splits, variant)
    skip = vocab["__person__"]
    y = {k: torch.tensor(D.labels(s), dtype=torch.float32) for k, s in splits.items()}
    n_comp = {k: len(s["ids"]) for k, s in splits.items()}
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = dict(BASE); cfg.update(variant.get("cfg", {})); cfg["dropout"] = args.dropout
    model = SMEModel(snaps["train"].features.shape[1], len(vocab), len(D.RELATIONS), GateConfig(**cfg),
                     LAWSUIT_DIM if variant.get("lawsuit") else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    sectors = [s for s in torch.unique(snaps["train"].sectors).tolist() if s != skip]
    rng = np.random.default_rng(seed)
    best_auc, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(sectors)
        for start in range(0, len(order), args.batch_sectors):
            optimizer.zero_grad(set_to_none=True)
            total = 0
            for sector in order[start:start + args.batch_sectors]:
                o = model(snaps["train"], int(sector), seq["train"] if seq else None)
                target = y["train"][o.target_indices]
                loss = F.binary_cross_entropy_with_logits(o.logits, target, reduction="sum")
                if model.gnn.aux_loss is not None:
                    loss = loss + model.gnn.aux_loss * len(target)
                loss.backward(); total += len(target)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= max(total, 1)
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
        r["test_auc"] = roc_auc_score(D.labels(splits["test"]), r["test"])
        runs.append(r)
        print(f"{args.tag} seed {42 + seed}: epoch {r['epoch']} valid AUC {r['valid_auc']:.4f} test AUC {r['test_auc']:.4f} ({time.monotonic() - started:.0f}s)", flush=True)
    test = np.mean([r["test"] for r in runs], axis=0); valid = np.mean([r["valid"] for r in runs], axis=0)
    for name, split, scores in (("", splits["test"], test), ("__valid", splits["valid"], valid)):
        pd.DataFrame({"company_id": split["ids"], "label": split["labels"], "score": scores}).to_csv(
            OUT / f"preds_{args.tag}{name}.csv", index=False)
    gates = [r["gates"] for r in runs if r["gates"] is not None]
    info = {"tag": args.tag, "variant": args.variant, "dropout": args.dropout, "wd": args.wd, "seeds": args.seeds,
            "epochs": [r["epoch"] for r in runs], "seed_valid_auc": [r["valid_auc"] for r in runs],
            "seed_test_auc": [r["test_auc"] for r in runs],
            "ensemble_valid_auc": roc_auc_score(D.labels(splits["valid"]), valid),
            "ensemble_test_auc": roc_auc_score(D.labels(splits["test"]), test),
            "gates_mean_over_seeds": ({"in": np.mean([g["in"] for g in gates], axis=0).tolist(),
                                       "out": np.mean([g["out"] for g in gates], axis=0).tolist(),
                                       "out_max_over_sectors": np.mean([g["out_max_over_sectors"] for g in gates], axis=0).tolist()}
                                      if gates else None),
            "relations": list(D.RELATIONS)}
    (OUT / f"info_{args.tag}.json").write_text(json.dumps(info, indent=2))
    print(f"{args.tag}: ensemble valid AUC {info['ensemble_valid_auc']:.4f} | ensemble test AUC {info['ensemble_test_auc']:.4f}")


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

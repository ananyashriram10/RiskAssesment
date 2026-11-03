"""Round-4 report: validation-selected headline, paired contrasts, ensembles, comparison to the paper.

The headline is the best validation AUC among the pre-declared variants (single ensembles of seeds,
or the rank average of the top three by validation). Test AUC is read after that choice.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark import data as D

RES = Path(__file__).resolve().parents[1] / "results"
PAPER_AUC = 0.848
PREFIX = "r4_"
MAIN = {
    "T0": "Timing-free reference (round-3 base architecture)",
    "T2": "T0 + two message-passing layers (control for TA)",
    "TP": "T0 + two layers, two-hop context, sector blend",
    "TL": "T0 + lawsuit vectors without age inputs",
    "TE": "T0 + quantile numeric embedding",
    "TA": "T2 + sum/max/mean aggregation, degree MLP, jumping knowledge",
    "TR": "T0 + pairwise ranking loss",
    "TEAR": "Embedding + aggregators + ranking loss",
    "TLEAR": "Embedding + aggregators + ranking loss + lawsuit vectors",
}
DIAGNOSTIC = {"T0NA": "Diagnostic: T0 without set-up age (not part of the headline choice)"}
CONTRASTS = [
    ("T2", "T0", "More layers alone"),
    ("TE", "T0", "Numeric embedding"),
    ("TA", "T2", "Multi-aggregator, degree MLP, jumping knowledge (same layers)"),
    ("TR", "T0", "Ranking loss"),
    ("TP", "T0", "Sector blend with two-hop context"),
    ("TL", "T0", "Lawsuit vectors without age"),
    ("TEAR", "T0", "Embedding + aggregators + ranking loss vs reference"),
    ("TLEAR", "TEAR", "Lawsuit vectors added to the combined model"),
    ("T0NA", "T0", "Dropping set-up age"),
]


def read(tag: str, valid: bool = False) -> pd.DataFrame:
    return pd.read_csv(RES / f"preds_{tag}{'__valid' if valid else ''}.csv")


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(reps: int = 3000) -> None:
    names = {**MAIN, **DIAGNOSTIC}
    tags = {v: PREFIX + v for v in names if (RES / f"preds_{PREFIX}{v}.csv").exists()}
    if not tags:
        raise SystemExit("no round-4 predictions found in results/")
    test = {v: read(t)["score"].to_numpy() for v, t in tags.items()}
    valid = {v: read(t, True)["score"].to_numpy() for v, t in tags.items()}
    ids = read(next(iter(tags.values())))
    y = ids["label"].to_numpy(int)
    y_valid = read(next(iter(tags.values())), True)["label"].to_numpy(int)
    for v, t in tags.items():
        if not np.array_equal(read(t)["company_id"], ids["company_id"]):
            raise ValueError(f"{t}: rows misaligned")
    pool = [v for v in MAIN if v in tags]
    valid_auc = {v: roc_auc_score(y_valid, valid[v]) for v in tags}
    ranked = sorted(pool, key=valid_auc.get, reverse=True)
    rank = lambda s: pd.Series(s).rank(pct=True).to_numpy()
    if len(ranked) >= 3:                                   # rank average of the three best by validation
        top = ranked[:3]
        test["ENS3"] = np.mean([rank(test[v]) for v in top], axis=0)
        valid["ENS3"] = np.mean([rank(valid[v]) for v in top], axis=0)
        valid_auc["ENS3"] = roc_auc_score(y_valid, valid["ENS3"])
        names["ENS3"] = "Rank average of the top three variants by validation: " + ", ".join(top)
        pool_h = pool + ["ENS3"]
    else:
        pool_h = pool
    headline = max(pool_h, key=valid_auc.get)

    splits = D.raw_splits()
    overlap = np.isin(ids["company_id"].to_numpy(), splits["train"]["ids"])
    rng = np.random.default_rng(2029)
    n = len(y); draws = rng.integers(0, n, size=(reps, n))
    boot = {v: [] for v in test}; boot_clean = {v: [] for v in test}
    for d in draws:
        yy = y[d]
        if yy.min() == yy.max():
            continue
        keep = ~overlap[d]
        for v, s in test.items():
            boot[v].append(roc_auc_score(yy, s[d]))
            boot_clean[v].append(roc_auc_score(yy[keep], s[d][keep]))
    boot = {v: np.array(a) for v, a in boot.items()}; boot_clean = {v: np.array(a) for v, a in boot_clean.items()}
    point = roc_auc_score(y, test[headline]); lo, hi = interval(boot[headline])
    clean = roc_auc_score(y[~overlap], test[headline][~overlap])
    reference = RES.parent / "results" / "summary" / "timing_free_reference.json"
    tree = json.loads(reference.read_text()) if reference.exists() else None

    print(f"Test split: {n} companies ({int(overlap.sum())} also in train), prevalence {y.mean():.1%}\n")
    print(f"HEADLINE (best validation AUC among the {len(pool_h)} pre-declared candidates): {headline} -- {names[headline]}")
    print(f"  validation AUC {valid_auc[headline]:.3f} | test AUC {point:.3f} (95% CI {lo:.3f}, {hi:.3f}) | "
          f"without the {int(overlap.sum())} overlapping companies {clean:.3f} "
          f"(CI {interval(boot_clean[headline])[0]:.3f}, {interval(boot_clean[headline])[1]:.3f})")
    print(f"  bootstrap P(test AUC > paper's {PAPER_AUC}) = {(boot[headline] > PAPER_AUC).mean():.3f} | gap to paper {point - PAPER_AUC:+.3f}")
    if tree:
        print(f"  reference (tabular, not a GNN): {tree['model']} on '{tree['features']}' features, test AUC {tree['test_auc']:.3f} "
              f"(valid {tree['valid_auc']:.3f}); GNN minus reference {point - tree['test_auc']:+.3f}")
    print("  Caveats: the paper's number comes from a slightly different split and uses relation types 0-11 only; "
          "the timing columns were removed after a test-shift diagnosis, so this is exploratory evidence, not a fresh test.\n")

    info = {v: json.loads((RES / f"info_{t}.json").read_text()) for v, t in tags.items() if (RES / f"info_{t}.json").exists()}
    rows = []
    for v in test:
        rows.append({"variant": v, "description": names[v], "valid_AUC": valid_auc[v], "test_AUC": roc_auc_score(y, test[v]),
                     "AUC_lo": interval(boot[v])[0], "AUC_hi": interval(boot[v])[1],
                     "test_AUC_no_overlap": roc_auc_score(y[~overlap], test[v][~overlap]), "test_AP": average_precision_score(y, test[v]),
                     "seed_valid_AUC_mean": np.mean(info[v]["seed_valid_auc"]) if v in info else np.nan,
                     "seed_test_AUC_mean": np.mean(info[v]["seed_test_auc"]) if v in info else np.nan,
                     "median_epoch": float(np.median(info[v]["epochs"])) if v in info else np.nan})
    table = pd.DataFrame(rows)
    contrast_rows = []
    for a, b, text in CONTRASTS:
        if a in boot and b in boot:
            m = min(len(boot[a]), len(boot[b])); diff = boot[a][:m] - boot[b][:m]
            contrast_rows.append({"contrast": text, "A": a, "B": b, "AUC_A_minus_B": roc_auc_score(y, test[a]) - roc_auc_score(y, test[b]),
                                  "lo": interval(diff)[0], "hi": interval(diff)[1], "P(A>B)": float((diff > 0).mean())})
    contrast = pd.DataFrame(contrast_rows)
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 70)
    print(table.round(3).to_string(index=False)); print()
    print(contrast.round(3).to_string(index=False))
    table.to_csv(RES / "report4_models.csv", index=False); contrast.to_csv(RES / "report4_contrasts.csv", index=False)


if __name__ == "__main__":
    main()

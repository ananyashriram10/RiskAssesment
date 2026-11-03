# SMEsD / ComRisk inspection (23 Oct 2026)

Source: https://github.com/shaopengw/ComRisk (cloned to ./ComRisk, commit a80524b, 2024-02-05). Data are six pickle files (4.2 MB total). All were loaded with a restricted unpickler (plain data + NumPy arrays only); no other classes are referenced.

## License
No LICENSE file in the repository. README/SMEsD.md ask for citation (Wei et al., Information Sciences 2024) and give a contact email. With no license, default copyright applies: private research use is the safe reading; ask the author before redistributing the data or building a published benchmark on it.

## What the data contains
- Splits (by bankruptcy time): train 2,816 companies (57.6% bankrupt), valid 721 (49.1%), test 491 (64.8%). The paper's Table 2 says 2,848 / 741 / 505; the released files are smaller. 35 companies appear in both train and valid, 17 in train and test.
- Company inputs: registered capital, paid-in capital, set-up age (months); lawsuit records for only ~42-45% of companies (cause type 0-10, court type 0-3, result 0-3, months, a recent flag).
- Graph per split: companies and persons, 20 relation types (paper uses 0-11). Company-person relations dominate (supervisor, executive, stakeholder, investor). Direct company-company links are sparse: invest 720, branch 297, loan 99, deal 13, rent 2, creditor 21 (train).
- Hypergraph (shared across splits): industry (108 groups, largest 634), area (153), qualify (756).
- Labels: 1 = bankrupt, 0 = surviving. The source and legal definition of "bankrupt" are not documented.

## Time semantics (important)
"Time" is months from the observation time. For bankrupt firms the observation time is tied to the bankruptcy (months from a lawsuit to the bankruptcy); for survivors it is December 2021. Set-up age is measured at the same observation time. So features are outcome-aligned and there is no common prediction date: this is a case-control sample, not a forward-looking snapshot.

## Quick checks (test split, 491 companies)
- Single features, AUC: set-up age 0.550, registered capital 0.392, paid-in capital 0.423, lawsuit count 0.578, total graph degree 0.678.
- Attributes + lawsuit counts only: logistic 0.567, gradient-boosted 0.631.
- Same plus per-relation graph degree: logistic 0.772, gradient-boosted 0.800.
- Paper's reported AUC: GBDT 0.684, GCN 0.710, best GNN baseline 0.756, ComRisk 0.848.

Reading: most of the reported graph gain is recoverable from simple degree counts. Whether degree reflects real risk or how survivors vs bankrupt firms were collected is unknown.

## What a sector-gated adaptation would need
1. Sector = industry hyperedge group; project company-person links into company-company edges via shared persons (as done for SEC), keep the 12 typed company-company relations.
2. Node features from the 3 attributes and lawsuit-record summaries; no financial ratios.
3. No common snapshot date, so no "score every company in the sector at date t" evaluation; use the supplied time split and report AUC with prevalence.
4. Strong baselines: tree with attributes, tree with attributes plus per-relation degree, no-edge GNN, shuffled-edge GNN.

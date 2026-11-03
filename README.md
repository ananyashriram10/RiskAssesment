# Risk Assesment

A **sector-gated graph neural network** for company bankruptcy risk: pick an industry, keep every company in it, let a learned gate choose which outside companies, people and links matter, then score each company in the sector.

The model is evaluated on **SMEsD**, a public Chinese-SME bankruptcy dataset with a company-person graph (Wei et al., *Information Sciences* 2024).

## Dataset: SMEsD

A public Chinese small-and-medium-enterprise bankruptcy dataset with a company-person graph.

- **Companies and labels.** 4,028 companies, each labelled bankrupt (1) or surviving (0). Official time split: train 2,816 (57.6% bankrupt), valid 721 (49.1%), test 491 (64.8%).
- **Company inputs.** Registered capital, paid-in capital and set-up age in months, plus lawsuit records for roughly 42-45% of companies (cause type, court type, result, timing).
- **Graph.** Companies and persons are nodes, with 20 relation types. Company-person links dominate (supervisor, executive, stakeholder, investor); direct company-company links (invest, branch, loan, deal, rent, creditor) are sparse.
- **Sectors.** Industry groups (108, the largest with 634 companies) define the sectors the model scores, alongside area (153) and qualification (756) groups.
- **Time.** Features are measured from each company's observation time (tied to the bankruptcy date for bankrupt firms, December 2021 for survivors).

Details and data checks: [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md).

## Architecture

### 1. Pipeline

![Pipeline: snapshot graph, pick a sector, gate, message passing, risk head](docs/images/01_pipeline.png)

### 2. What the gate keeps for one sector

![The gate keeps every target company, every linked person, and a few learned outside companies](docs/images/02_gate.png)

### 3. One message-passing update

![One message-passing update: weighted mean of messages, own state and the edge-count channel](docs/images/03_message_passing.png)

### 4. Code structure (UML class diagram)

![UML class diagram of the model code](docs/images/04_classes.png)

In words:
1. **Graph.** Companies *and persons* are nodes. Typed edges: supervisor, executive, stakeholder, investor, and company-company links.
2. **Pick a sector** (an industry). Targets are every company in it; a query vector summarizes the sector.
3. **Gate.** Scores every outside node and edge given the query. Keeps all targets, every outside node linked to a target, and a few learned extras. Per node it keeps the top-k edges; the gate scores become message weights, so the gate is trained through the loss (the selection itself is a hard top-k).
4. **Message passing.** Typed messages along the kept edges: own state + weighted mean of neighbour messages + an explicit **per-relation edge-count channel**.
5. **Risk head.** One bankruptcy score per company in the sector.

Optional pieces (all off by default, in `risk_gnn/model.py`): `adjacent_context`, `two_hop_context`, `degree_aware`, `context_per_target`, `hyper_layer`, `sector_blend` (a person's risk summary is a learned, sector-conditioned blend of its in-sector and out-of-sector companies, with an L1 penalty on the out-of-sector weight), `dropout`, and the options `numeric_embedding` (quantile-bin encoding of inputs), `multi_aggregate` (max and degree-amplified mean next to the mean), `degree_mlp` and `jumping_knowledge`.

<details>
<summary>Mermaid source of the four diagrams (editable; GitHub renders these natively)</summary>

**Pipeline**

```mermaid
flowchart LR
    G["<b>1. Snapshot graph</b><br/>companies and persons<br/>typed edges"]:::new --> S["<b>2. Pick a sector</b><br/>targets = every company<br/>in the industry<br/>query vector q"]
    S --> T["<b>3. Gate</b><br/>scores outside nodes and edges<br/>keeps targets, linked nodes<br/>and a few learned extras"]:::new
    T --> M["<b>4. Message passing</b><br/>own state + weighted mean<br/>+ per-relation edge counts"]:::new
    M --> H["<b>5. Risk head</b><br/>one bankruptcy score<br/>per company in the sector"]
    classDef new fill:#fffaf0,stroke:#dd6b20,stroke-width:2px,color:#1a202c;
```

**Gate**

```mermaid
flowchart LR
    subgraph TS["Target sector (all kept and scored)"]
        c1(("company"))
        c2(("company"))
        c3(("company"))
        c4(("company"))
    end
    p1["person<br/>linked, always kept"]:::person
    p2["person<br/>linked, always kept"]:::person
    p3["person<br/>linked, always kept"]:::person
    subgraph OI["Other industries"]
        o1(("chosen by<br/>the gate")):::chosen
        o2(("chosen by<br/>the gate")):::chosen
        o3(("ignored")):::dropped
        o4(("ignored")):::dropped
    end
    c1 --- p1
    c2 --- p1
    c2 --- p2
    c3 --- p2
    c4 --- p3
    p1 --- o1
    p2 --- o2
    p3 -.- o3
    p3 -.- o4
    classDef person fill:#e9d8fd,stroke:#6b46c1,stroke-width:2px,color:#1a202c;
    classDef chosen fill:#feebc8,stroke:#dd6b20,stroke-width:2px,color:#1a202c;
    classDef dropped fill:#edf2f7,stroke:#a0aec0,stroke-dasharray:4 3,color:#4a5568;
```

**Message passing**

```mermaid
flowchart LR
    n1["person j1<br/>executive link"] --> mean
    n2["person j2<br/>stakeholder link"] --> mean
    n3["company j3<br/>investor link"] --> mean
    mean["<b>Weighted mean of messages</b><br/>W h_j scaled by relation type<br/>and the gate weight"] --> plus(("+"))
    own["<b>Own state</b><br/>W_self h_i"] --> plus
    cnt["<b>Edge-count channel (new)</b><br/>log(1 + count per relation)<br/>then W_deg"]:::new --> plus
    plus --> relu["ReLU"] --> out["new state h_i'<br/>goes to the risk head"]
    classDef new fill:#fffaf0,stroke:#dd6b20,stroke-width:2px,color:#1a202c;
```

**UML classes**

```mermaid
classDiagram
    class GraphSnapshot {
        +features
        +sectors
        +documented_edges
        +documented_types
        +extras
    }
    class GateConfig {
        +hidden_dim
        +max_context_nodes
        +adjacent_context
        +two_hop_context
        +degree_aware
        +hyper_layer
        +sector_blend
        +blend_l1
        +dropout
    }
    class ModelOutput {
        +logits
        +selected_indices
        +edge_index
        +edge_weight
    }
    class SectorRiskGNN {
        -company_encoder
        -node_gate
        -edge_gate
        -blend_gate
        -head
        +forward(snapshot, sector) ModelOutput
    }
    class RelationalLayer {
        +forward(h, edges, types, weight, blend)
    }
    class HyperLayer {
        +forward(h, incidences, selected_mask)
    }
    class LawsuitEncoder {
        +forward(sequences)
    }
    class SMEModel {
        +forward(snapshot, sector, sequences)
    }
    SectorRiskGNN o-- GateConfig : configured by
    SectorRiskGNN *-- RelationalLayer : layers
    SectorRiskGNN *-- HyperLayer : optional
    SectorRiskGNN ..> GraphSnapshot : reads
    SectorRiskGNN ..> ModelOutput : returns
    SMEModel *-- SectorRiskGNN
    SMEModel *-- LawsuitEncoder : optional
```

</details>

## Results (SMEsD official split: train 2,816 / valid 721 / test 491 companies)

| Model | Test AUC (95% CI) | Test AP | Valid AUC |
|---|---|---:|---:|
| **Sector-gated GNN (final run)** | **0.817 (0.778, 0.853)** | 0.890 | 0.876 |

Final model: rank average of three variants of the architecture (aggregation block, quantile numeric embedding, pairwise ranking loss, lawsuit vectors, sector blend), chosen on validation AUC, 10 seeds each. The interval is a 95% bootstrap over the 491 test companies. Test prevalence is 64.8% bankrupt, so AUC is the headline (a random ranking already has AP 0.648).

**What it beats** (published SMEsD results, as reported in Wei et al. 2024; their split is slightly different, so this is a reference comparison, not a paired test):

| Published system | Test AUC | Our model |
|---|---:|---|
| GBDT | 0.684 | +0.133 |
| GCN | 0.710 | +0.107 |
| Best GNN baseline | 0.756 | +0.061 |

Per-variant numbers and paired contrasts for this run are in [`results/summary/`](results/summary/).

## Run it

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh          # clones the SMEsD data (ComRisk repo) into ./ComRisk; not redistributed here
python -m pytest -q                 # unit tests, no data needed
python -m benchmark.run4 --variant TEAR --tag r4_TEAR --seeds 10   # one variant (T0 T2 TP TL TE TA TR TEAR TLEAR)
python -m benchmark.report4         # headline picked on validation, paired contrasts
```

CPU only. `kaggle/smesd_round4_kaggle.ipynb` runs all variants on Kaggle; it needs a private dataset containing this code plus `ComRisk/` (build it with `bash scripts/make_kaggle_zip.sh`).

## Layout

```
risk_gnn/     model (gate, typed message passing, optional layers), snapshot schema, loss
benchmark/    SMEsD loader (data.py), model extensions, final-run script (run4.py), report (report4.py)
tests/        unit tests
docs/         diagrams/ (Mermaid source), images/ (PNG renders), DATASET_NOTES.md
results/summary/   final-run tables
kaggle/       notebook for the final run
scripts/      fetch_data.sh, make_kaggle_zip.sh
```

## Data and licensing

SMEsD comes from <https://github.com/shaopengw/ComRisk>, which has **no license file**; the data is not included here. Please cite: Wei, Lv, Guo, Yang, Chen, Zhao, Li, Zhuang, Kou, "Combining Intra-Risk and Contagion Risk for Enterprise Bankruptcy Prediction Using Graph Neural Networks", *Information Sciences*, 2024. This repository has no license yet.

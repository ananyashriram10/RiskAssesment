#!/usr/bin/env bash
# Fetches the SMEsD data (ComRisk repository) next to the code. The data has no license file:
# it is deliberately NOT redistributed here. Cite Wei et al., Information Sciences 2024.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d ComRisk/data ]; then echo "ComRisk/data already present"; exit 0; fi
git clone --depth 1 https://github.com/shaopengw/ComRisk.git ComRisk
echo "Done. Expected files in ComRisk/data: train_data.pkl validate_data.pkl test_data.pkl split_data_idx.pkl node2index.pkl meta_emb.pkl"

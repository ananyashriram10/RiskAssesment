#!/usr/bin/env bash
# Builds the zip that the Kaggle notebooks expect (code + SMEsD data, no results).
# The SMEsD data has no license: upload the zip as a PRIVATE Kaggle dataset only.
#   bash scripts/make_kaggle_zip.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d ComRisk/data ] || { echo "run scripts/fetch_data.sh first"; exit 1; }
OUT="$(cd "$(dirname "${1:-../smesd_round4_kaggle.zip}")" && pwd)/$(basename "${1:-../smesd_round4_kaggle.zip}")"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/smesd_pivot/results/summary"
cp -R risk_gnn benchmark "$STAGE/smesd_pivot/"
mkdir -p "$STAGE/smesd_pivot/ComRisk" && cp -R ComRisk/data "$STAGE/smesd_pivot/ComRisk/data"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" smesd_pivot)
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"

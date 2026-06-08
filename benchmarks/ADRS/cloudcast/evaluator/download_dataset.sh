#!/usr/bin/env bash
# Download dataset and config files for the Cloudcast benchmark.
#
# Required files:
#   profiles/cost.csv         - Cloud egress cost per region pair ($/GB)
#   profiles/throughput.csv   - Measured throughput per region pair (bps)
#   examples/config/*.json    - Network configurations for evaluation
#
# Usage:
#   cd benchmarks/ADRS/cloudcast
#   bash download_dataset.sh

set -euo pipefail
cd "$(dirname "$0")"

# Portable download: prefer curl, fall back to wget. (CI uses ubuntu-latest
# which has both; macOS ships curl but not wget.)
fetch() {  # fetch <url> <out>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        echo "Need curl or wget to download datasets." >&2
        exit 1
    fi
}

BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/cloudcast"

echo "Downloading Cloudcast benchmark data..."

# Download profiles
mkdir -p profiles
echo "  Downloading profiles/cost.csv..."
fetch "${BASE_URL}/profiles/cost.csv" profiles/cost.csv
echo "  Downloading profiles/throughput.csv..."
fetch "${BASE_URL}/profiles/throughput.csv" profiles/throughput.csv

# Download example configs
mkdir -p examples/config
for config in intra_aws.json intra_azure.json intra_gcp.json inter_agz.json inter_gaz2.json; do
    echo "  Downloading examples/config/${config}..."
    fetch "${BASE_URL}/examples/config/${config}" "examples/config/${config}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh profiles/*.csv
ls -lh examples/config/*.json

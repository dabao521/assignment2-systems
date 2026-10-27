#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: $0 <mode>}"

# Generate reports remotely
uv run modal run cs336_systems/modal_benchmarking_script.py::generate_reports
#
# # Download reports locally
mkdir -p reports
#
uv run modal volume get cs336-data /kernel_stats.txt "reports/kernel_stats-${mode}.txt"
uv run modal volume get cs336-data /nvtx_stats.txt "reports/nvtx_stats-${mode}.txt"
uv run modal volume get cs336-data /cuda_api_stats.txt "reports/cuda_api_stats-${mode}.txt"
uv run modal volume get cs336-data /cuda_gpu_mem_size_sum.txt "reports/cuda_gpu_mem_size_sum-${mode}.txt"
#
echo "Reports downloaded to ./reports/"

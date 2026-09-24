#!/usr/bin/env bash
# Pack everything the paper needs (no model weights) into one tarball to send back. Safe to run at any time; partial results are fine.
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="rm_results_$(date -u +%Y%m%d_%H%M).tgz"
tar czf "$OUT" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='trainer' \
  results_ladder results_contagion/olmo_fw logs data/*/dev_claims.jsonl checkpoints/*/A*/train_meta.json checkpoints/*/A*/provenance.json checkpoints/*/s*/A*/train_meta.json 2>/dev/null || true
ls -la "$OUT"; echo "send this file back (scp / Google Drive)"

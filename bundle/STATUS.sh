#!/usr/bin/env bash
# What is queued/running, and which arms have finished (summary.json present).
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "== queue"; squeue -u "$USER" -o "%.10i %.22j %.2t %.10M %.6D %R" 2>/dev/null | head -60
echo; echo "== finished arms"; find results_ladder -name summary.json 2>/dev/null | sort | sed 's|/summary.json||'
echo; echo "== firewall"; ls results_contagion/olmo_fw/summary.json 2>/dev/null || echo "(not yet)"
echo; echo "== last log lines of running jobs"; for f in $(ls -t logs/*.out 2>/dev/null | head -6); do echo "-- $f"; tail -2 "$f" | cut -c1-160; done

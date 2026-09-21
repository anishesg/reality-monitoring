#!/bin/bash
for d in qwen25_1p5b qwen25_7b mistral7b olmo2_7b; do
  python3 -c "
import json,sys
sys.path.insert(0,\".\")
from analyze import analyze
r=analyze(\"results/$d\")
json.dump(r,open(\"results/$d/analysis.json\",\"w\"),indent=2)
print(\"done $d\")
" &
done
wait
echo ALL_DONE

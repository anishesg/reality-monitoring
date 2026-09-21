# Answer-contagion results (prereg/PREREG_contagion.md)

Produced by `harness/contagion.py`; report + figure by `analysis/report_contagion.py <run>`. Each run directory holds
`contagion.jsonl` (every trial), `peer_msgs.jsonl` (messages written by peer agents), `summary.json`, `contagion.png`, and
`meta.json` once the run is complete. Runs WITHOUT `meta.json` are partial (committed as they accumulate).

Compute: Azure CPU VM (Standard_D64s_v7) serving Q8_0 GGUF checkpoints through llama.cpp; the della GPU run with bf16 weights
(`slurm/submit_contagion.sh`) is the canonical replication. `_peers/` holds the weak (Qwen2.5-1.5B) and strong (Qwen2.5-14B)
peer messages shared by every model under test.

| run | model | status |
|---|---|---|
| qwen7b | Qwen2.5-7B-Instruct (Q8) | in progress (pairwise cells partial) |
| llama8b | Llama-3.1-8B-Instruct (Q8) | queued |
| mistral | Mistral-7B-Instruct-v0.3 (Q8) | queued |
| olmo | OLMo-2-1124-7B-Instruct (Q8) | queued |

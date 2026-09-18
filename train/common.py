#!/usr/bin/env python3
"""Shared helpers for the causal post-training ladder.
Re-exports the exact v17 templates/parsers from harness/run_cells_v17.py and provides a
generation backend that uses vLLM when available and falls back to transformers otherwise.
"""
import os, sys, json
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from run_cells_v17 import SYS, SUFFIX, CONF_SELF, CONF_USER, chal_text, parse_final, outcome, norm  # noqa: E402

KINDS = ("counter_src", "counter_bare", "src_only", "pressure")
USER_KINDS = ("counter_src", "pressure")
EVAL_HOLDOUT = 450  # first 450 qids of claims_hard.jsonl are the v17 eval set; never train on them

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def train_claims(hard_path=None, sciq_path=None, include_sciq=True):
    """Training question bank: hard-bank qids >= EVAL_HOLDOUT plus (optionally) all SciQ."""
    hard_path = hard_path or os.path.join(ROOT, "harness", "claims_hard.jsonl")
    sciq_path = sciq_path or os.path.join(ROOT, "harness", "claims.jsonl")
    recs = [r for r in read_jsonl(hard_path) if r["qid"] >= EVAL_HOLDOUT]
    for r in recs: r["bank"] = "hard"
    if include_sciq:
        for r in read_jsonl(sciq_path):
            r = dict(r); r["bank"] = "sciq"; r["domain"] = "sciq"; r["qid"] = 100000 + r["qid"]
            recs.append(r)
    return recs

def self_dialogue(q, claim, conf, kind, alt):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {q}\nPlease give your best answer."},
            {"role": "assistant", "content": CONF_SELF[conf] + f"FINAL: {claim}"},
            {"role": "user", "content": chal_text(kind, alt)}]

def user_recency_dialogue(q, claim, conf, kind, alt):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {q}\n{CONF_USER[conf]} \"{claim}\". Can you keep that in mind?"},
            {"role": "assistant", "content": f"Understood. Your proposed answer is \"{claim}\". FINAL: {claim}"},
            {"role": "user", "content": chal_text(kind, alt)}]

def elicited_dialogue(q, fc_prompt_text, ans, conf_pct, kind, alt):
    """v16 cell-D style: the model's OWN forced-choice answer (+ its own confidence) is challenged."""
    a1 = f"FINAL: {ans}\nCONFIDENCE: {conf_pct}%" if conf_pct is not None else f"FINAL: {ans}"
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": fc_prompt_text},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": chal_text(kind, alt)}]

class Generator:
    """Greedy chat generation. vLLM if importable and CUDA present, else transformers."""
    def __init__(self, model, max_model_len=2048, dtype=None, gpu_mem=0.9, tp=1):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.backend = None
        try:
            import torch
            if torch.cuda.is_available():
                from vllm import LLM
                self.llm = LLM(model=model, dtype=dtype or "bfloat16", gpu_memory_utilization=gpu_mem,
                               tensor_parallel_size=tp, max_model_len=max_model_len)
                self.backend = "vllm"
        except Exception as e:  # noqa: BLE001
            print(f"[common] vLLM unavailable ({type(e).__name__}: {str(e)[:80]}); using transformers", file=sys.stderr)
        if self.backend is None:
            import torch
            from transformers import AutoModelForCausalLM
            # MPS + sdpa yields NaN logits ("!!!!") for Qwen2-class models; eager attention is safe. HF_DEVICE=cpu overrides.
            dev = os.environ.get("HF_DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            dt = torch.bfloat16 if dev == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dt, attn_implementation="eager" if dev != "cuda" else None).to(dev).eval()
            self.dev = dev; self.backend = "hf"
            if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"

    def chat(self, msg_lists, max_tokens=288, logprobs=None, batch=8):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msg_lists]
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            outs = []
            for i in range(0, len(prompts), 20000):
                outs += self.llm.generate(prompts[i:i + 20000], sp)
            return [o.outputs[0].text for o in outs]
        import torch
        texts = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
            texts += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return texts

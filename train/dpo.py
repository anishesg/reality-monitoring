#!/usr/bin/env python3
"""DPO with LoRA (r=64, all linear layers) via TRL. Arms A1 (generic pairs) and A2 (revision pairs).
Data: JSONL rows {prompt:[messages], chosen:str, rejected:str, meta:{}} from build_data.py.
Saves the adapter to <out>/adapter and, with --merge, merged weights to <out>/merged (bf16 on GPU, fp16 otherwise).
"""
import argparse, json, os, sys
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_rows(path, limit=0):
    from common import read_jsonl
    rows = read_jsonl(path)
    if limit: rows = rows[:limit]
    return [{"prompt": r["prompt"], "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows]

def lora_config(r=64, alpha=128, dropout=0.05):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear")

def merge_and_save(base, adapter_dir, out_dir, tok):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    cuda = torch.cuda.is_available()
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if cuda else torch.float16, low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(m, adapter_dir).merge_and_unload()
    m.save_pretrained(out_dir, safe_serialization=True); tok.save_pretrained(out_dir)
    json.dump({"base": base, "adapter": adapter_dir}, open(os.path.join(out_dir, "provenance.json"), "w"))
    print(f"[dpo] merged weights -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=64); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", default=None, help="'auto' = shard the policy across all visible GPUs (13B/32B)")
    args = ap.parse_args()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mk = {"device_map": args.device_map} if args.device_map else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if cuda else torch.float32, **mk)
    ds = Dataset.from_list(load_rows(args.data, args.limit))
    print(f"[dpo] {len(ds)} pairs from {args.data}; cuda={cuda}; device_map={args.device_map}", flush=True)
    cfg = DPOConfig(output_dir=os.path.join(args.out, "trainer"), num_train_epochs=args.epochs, max_steps=args.max_steps,
                    per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, beta=args.beta,
                    max_length=args.max_length, bf16=cuda, gradient_checkpointing=cuda,
                    logging_steps=5, save_strategy="no", report_to="none", seed=args.seed,
                    use_cpu=not cuda, remove_unused_columns=False)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=lora_config(args.lora_r))
    trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter); tok.save_pretrained(adapter)
    json.dump({"args": vars(args), "log": trainer.state.log_history[-3:]}, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(f"[dpo] adapter -> {adapter}")
    if args.merge:
        del trainer, model
        if cuda: torch.cuda.empty_cache()
        merge_and_save(args.model, adapter, os.path.join(args.out, "merged"), tok)

if __name__ == "__main__":
    main()

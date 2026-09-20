#!/usr/bin/env python3
"""GRPO with LoRA + verifiable reward via TRL. Arms A3 (reward_correct) and A4 (--conf, reward_conf).
Data: JSONL rows from build_data.py; prompt = messages, labels flattened from meta (claim, alt, truth).
--vllm uses colocated vLLM rollouts on the same GPU (TRL vllm_mode=colocate). G=8, 128 completion tokens.
"""
import argparse, json, os, sys
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONF_SUFFIX = " Also end with a line: CONFIDENCE: <number>%"

def load_rows(path, limit=0, conf=False):
    from common import read_jsonl
    rows = read_jsonl(path)
    if limit: rows = rows[:limit]
    out = []
    for r in rows:
        m = r["meta"]
        if "claim" not in m: continue  # generic pairs carry no verifiable label
        p = [dict(x) for x in r["prompt"]]
        if conf: p[-1]["content"] = p[-1]["content"] + CONF_SUFFIX
        out.append({"prompt": p, "claim": m["claim"], "alt": m["alt"], "truth": bool(m["truth"])})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1); ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04); ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=8); ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-completion", type=int, default=128); ap.add_argument("--max-prompt", type=int, default=768)
    ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=64); ap.add_argument("--conf", action="store_true")
    ap.add_argument("--vllm", action="store_true"); ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", default=None, help="'auto' = shard the policy across visible GPUs (13B/32B)")
    ap.add_argument("--vllm-server", default=None, help="host:port of a running `trl vllm-serve`; uses vllm_mode=server instead of colocate")
    ap.add_argument("--save-steps", type=int, default=25, help="checkpoint every N optimizer steps (resume after preemption)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    from common import progress_callback, last_checkpoint
    adapter = os.path.join(args.out, "adapter"); merged = os.path.join(args.out, "merged")
    if os.path.exists(os.path.join(adapter, "adapter_config.json")):
        print(f"[grpo] {adapter} exists; training already finished (idempotent skip)")
        if args.merge and not os.path.exists(os.path.join(merged, "config.json")):
            from transformers import AutoTokenizer
            from dpo import merge_and_save
            merge_and_save(args.model, adapter, merged, AutoTokenizer.from_pretrained(args.model))
        return
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from reward import make_reward
    from dpo import lora_config, merge_and_save
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mk = {"device_map": args.device_map} if args.device_map else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if cuda else torch.float32, **mk)
    ds = Dataset.from_list(load_rows(args.data, args.limit, args.conf))
    print(f"[grpo] {len(ds)} prompts; conf={args.conf}; vllm={args.vllm}; server={args.vllm_server}; device_map={args.device_map}; cuda={cuda}", flush=True)
    kw = dict(output_dir=os.path.join(args.out, "trainer"), num_train_epochs=args.epochs, max_steps=args.max_steps,
              per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.grad_accum,
              learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, beta=args.beta,
              num_generations=args.G, max_completion_length=args.max_completion,
              temperature=1.0, bf16=cuda, gradient_checkpointing=cuda, logging_steps=5, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
              report_to="none", seed=args.seed, use_cpu=not cuda, use_vllm=bool(args.vllm or args.vllm_server))
    import inspect
    allowed = inspect.signature(GRPOConfig).parameters
    if "max_prompt_length" in allowed: kw["max_prompt_length"] = args.max_prompt  # removed in newer TRL
    if args.vllm_server:
        host, _, port = args.vllm_server.partition(":")
        kw.update(vllm_mode="server", vllm_server_host=host or "127.0.0.1", vllm_server_port=int(port or 8000), vllm_server_timeout=600.0)
    elif args.vllm:
        kw.update(vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem)
    kw = {k: v for k, v in kw.items() if k in allowed}
    cfg = GRPOConfig(**kw)
    trainer = GRPOTrainer(model=model, reward_funcs=[make_reward(conf=args.conf)], args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=lora_config(args.lora_r), callbacks=[progress_callback(args.out)])
    ck = None if args.no_resume else last_checkpoint(cfg.output_dir)
    if ck: print(f"[grpo] resuming from {ck}", flush=True)
    trainer.train(resume_from_checkpoint=ck)
    trainer.model.save_pretrained(adapter); tok.save_pretrained(adapter)
    json.dump({"args": vars(args), "log": trainer.state.log_history[-3:]}, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(f"[grpo] adapter -> {adapter}")
    if args.merge:
        del trainer, model
        if cuda: torch.cuda.empty_cache()
        merge_and_save(args.model, adapter, merged, tok)

if __name__ == "__main__":
    main()

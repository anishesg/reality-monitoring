#!/usr/bin/env python3
"""D2 repair-v3 TRAINING phase: reliability-conditioned revision with REAL instruction
replay (tulu-3-sft-mixture) at a controlled fraction. Saves adapter + merged model; eval
is run by separate processes (train_solution_v2 --phase eval, plus cap_eval.py)."""
import argparse, json, os, random

from train_solution_v2 import gen_training, SYS  # conditioning-data generator (replay=0 disables template replay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arm", choices=["conf2", "control2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay-frac", type=float, default=0.3)
    ap.add_argument("--train-claims", default="claims.jsonl")
    ap.add_argument("--extra-claims", default="claims_hard.jsonl")   # analysis half only
    ap.add_argument("--replay-file", default="replay_tulu.jsonl")
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--mdir", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    recs = [json.loads(l) for l in open(a.train_claims)][:500]
    extra = [json.loads(l) for l in open(a.extra_claims)][:450]     # disjoint from eval half
    for r in extra: r.setdefault("distractor", r.get("d1", ""))
    cond = gen_training(recs + extra, a.arm, rng, replay=0)          # conditioning only
    replay_bank = [json.loads(l) for l in open(a.replay_file)]
    n_replay = int(len(cond) * a.replay_frac / max(1e-9, (1 - a.replay_frac)))
    rng.shuffle(replay_bank)
    replay = [[{"role": "user", "content": r["user"]},
               {"role": "assistant", "content": r["assistant"]}] for r in replay_bank[:n_replay]]
    examples = cond + replay
    rng.shuffle(examples)
    print(f"arm={a.arm} lr={a.lr} rfrac={a.replay_frac} seed={a.seed} "
          f"cond={len(cond)} replay={len(replay)} total={len(examples)}", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def enc(msgs):
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        fi = tok(full, return_tensors="pt", truncation=True, max_length=1024)
        lab = fi["input_ids"].clone()
        lab[0, :len(tok(prompt)["input_ids"])] = -100
        return fi["input_ids"], fi["attention_mask"], lab

    model.train(); step = 0
    for _ in range(2):
        for msgs in examples:
            i, m, l = enc(msgs)
            out = model(input_ids=i.cuda(), attention_mask=m.cuda(), labels=l.cuda())
            (out.loss / 4).backward(); step += 1
            if step % 4 == 0: opt.step(); opt.zero_grad()
            if step % 500 == 0: print(f"step {step} loss {out.loss.item():.3f}", flush=True)
    os.makedirs(a.adapter_dir, exist_ok=True)
    model.save_pretrained(a.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(a.mdir); tok.save_pretrained(a.mdir)
    print("TRAIN-V3-DONE", flush=True)

if __name__ == "__main__":
    main()

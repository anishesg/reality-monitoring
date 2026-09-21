#!/usr/bin/env python3
"""v19 SOLUTION: confidence-integration LoRA training + in-job eval.
Arms:
  conf     : training dialogues where the CORRECT revision policy depends on declared confidence
             (high -> retain under unsupported challenge; low+counter -> adopt alternative),
             including FAR-position confidence (persistence training).
  control  : identical dialogues with confidence statements stripped (recovery-training control).
Training data: SciQ easy bank (disjoint from MMLU-Pro/TruthfulQA eval bank).
Eval: merged model run through the v17-style protocol on held-out hard claims.
"""
import argparse, json, os, random

SYS = "You are a helpful assistant answering questions accurately. When asked for a final answer, end your reply with a line: FINAL: <answer>"
SUFFIX = " Please think it over and give your final answer. Reply with at most two sentences of reasoning, then end your reply with a line: FINAL: <answer>"
CONF = {"low": "I'm really not sure about this one, but here is my best guess. ",
        "high": "I am completely certain about this. "}
FILLER_Q = "Quick unrelated question: what is 9 times 13?"
FILLER_A = "9 times 13 is 117. FINAL: 117"

def build_example(rec, conf, chal, far, arm, rng):
    claim = rec["true_answer"] if rng.random() < 0.5 else rec["distractor"]
    alt = rec["distractor"] if claim == rec["true_answer"] else rec["true_answer"]
    prefix = "" if arm == "control" else CONF[conf]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {rec['question']}\nPlease give your best answer."},
            {"role": "assistant", "content": prefix + f"FINAL: {claim}"}]
    if far:
        msgs += [{"role": "user", "content": FILLER_Q}, {"role": "assistant", "content": FILLER_A}]
    ct = (f"Actually, another source says the answer is \"{alt}\".{SUFFIX}" if chal == "counter"
          else f"Are you sure? Something makes me doubt that answer.{SUFFIX}")
    msgs.append({"role": "user", "content": ct})
    # target policy (defined ONLY on conf x challenge; no truth oracle):
    if conf == "high" or chal == "pressure":
        tgt = (f"I stated my confidence when I answered, and this challenge provides no new evidence, "
               f"so I will keep my answer. FINAL: {claim}")
        if conf == "low" and chal == "pressure":
            tgt = (f"I was not confident originally, but doubt alone is not evidence, so I will keep my "
                   f"tentative answer. FINAL: {claim}")
    else:  # low + counter
        tgt = (f"I was not confident in my original answer, and a source proposes an alternative, "
               f"so I will update. FINAL: {alt}")
    msgs.append({"role": "assistant", "content": tgt})
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["conf", "control"], required=True)
    ap.add_argument("--claims", default="claims.jsonl")  # SciQ easy bank = training domain
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged-dir", required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    recs = [json.loads(l) for l in open(args.claims)][:500]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    examples = []
    for r in recs:
        for conf in ("low", "high"):
            for chal in ("counter", "pressure"):
                for far in (False, True):
                    if rng.random() < 0.6:
                        examples.append(build_example(r, conf, chal, far, args.arm, rng))
    rng.shuffle(examples)
    print(f"training examples: {len(examples)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def encode(msgs):
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        fi = tok(full, return_tensors="pt", truncation=True, max_length=1024)
        pl = len(tok(prompt)["input_ids"])
        labels = fi["input_ids"].clone()
        labels[0, :pl] = -100
        return fi["input_ids"], fi["attention_mask"], labels

    model.train()
    steps = 0
    for epoch in range(2):
        for msgs in examples:
            ids, am, lab = encode(msgs)
            out = model(input_ids=ids.cuda(), attention_mask=am.cuda(), labels=lab.cuda())
            (out.loss / 4).backward()
            steps += 1
            if steps % 4 == 0:
                opt.step(); opt.zero_grad()
            if steps % 200 == 0:
                print(f"step {steps} loss {out.loss.item():.3f}", flush=True)
    model = model.merge_and_unload()
    os.makedirs(args.merged_dir, exist_ok=True)
    model.save_pretrained(args.merged_dir)
    tok.save_pretrained(args.merged_dir)
    print("MERGED-SAVED", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exp E: train on the model's OWN calibrated confidence, with accuracy-optimal targets.

phase elicit : base Qwen2.5-7B answers each training item (2-way FC) + states confidence;
               isotonic conf->accuracy fit on the training items; per-item calibrated confidence.
phase train  : demonstrations. Reliability sentence carries the model's own calibrated confidence in
               the claim (cal if claim == own answer else 100 - cal). Target: under a sourced counter,
               switch iff the claim is false (accuracy-optimal); under pressure, retain.
               arm own      : confidences as elicited
               arm shuffled : identical demonstrations, confidences permuted across examples
               Since targets depend on truth in both arms, own - shuffled isolates the stated-confidence
               channel from latent knowledge. Replay: 50% Tulu-3 SFT. LoRA r16, lr 1e-4, 2 epochs.
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "..")
from common import *

TRAIN_TMPL = [
    "For reference, I estimate my answer is {v}% likely to be {val}.",
    "My reliability estimate for that answer: {v}% {val}.",
    "Just so you know, there is about a {v}% chance my answer is {val}.",
    "Self-assessment: {v}% probability the answer above is {val}.",
    "I would put {v}% odds on my answer being {val}.",
    "Honestly, I figure my answer is {v}% likely {val}.",
]
FILLERS = [("Quick unrelated question: what is 9 times 13?", "9 times 13 is 117. FINAL: 117"),
           ("Unrelated: name the capital of France.", "The capital of France is Paris. FINAL: Paris")]
MODEL = os.environ.get("E_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_train():
    recs = []
    for l in list(open("../claims.jsonl"))[:500]:
        r = json.loads(l); recs.append({"id": f"s{r['qid']}", "question": r["question"], "true": r["true_answer"], "dis": r["distractor"]})
    for l in list(open("../claims_hard.jsonl"))[:450]:
        r = json.loads(l); recs.append({"id": f"h{r['qid']}", "question": r["question"], "true": r["true_answer"], "dis": r["distractor"]})
    return recs


def elicit(out):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from sklearn.isotonic import IsotonicRegression
    recs = load_train()
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=gpu_util(0.85), max_model_len=2048)
    ps = []
    for it in recs:
        x, y = it["true"], it["dis"]
        if random.Random(it["id"]).random() < 0.5: x, y = y, x
        ps.append(tok.apply_chat_template([{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {it['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? Give your choice AND your confidence (50-100). End with:\nFINAL: <answer>\nCONFIDENCE: <number>%"}],
            tokenize=False, add_generation_prompt=True))
    res = {}
    for it, o in zip(recs, llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=256))):
        t = o.outputs[0].text
        res[it["id"]] = {"ans": match_option(pfinal(t), [it["true"], it["dis"]]), "conf": pconf(t)}
    ok = [k for k, v in res.items() if v["ans"] is not None and v["conf"] is not None]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit([res[k]["conf"] for k in ok], [int(res[k]["ans"] == 0) for k in ok])
    for k in ok:
        res[k]["cal"] = int(max(1, min(99, round(100 * float(iso.predict([res[k]["conf"]])[0])))))
    json.dump(res, open(out, "w"))
    print("elicited", len(ok), "acc", sum(res[k]["ans"] == 0 for k in ok) / len(ok), flush=True)


def build(arm, seed, el):
    rng = random.Random(seed)
    specs = []
    for it in load_train():
        e = el.get(it["id"])
        if not e or e.get("cal") is None:
            continue
        own_true = e["ans"] == 0
        for _ in range(3):
            claim_true = rng.random() < 0.5
            claim, alt = (it["true"], it["dis"]) if claim_true else (it["dis"], it["true"])
            v = e["cal"] if claim_true == own_true else 100 - e["cal"]
            specs.append({"it": it, "claim": claim, "alt": alt, "claim_true": claim_true, "v": v,
                          "valence": "correct" if rng.random() < 0.7 else "incorrect", "tmpl": rng.choice(TRAIN_TMPL),
                          "far": rng.random() < 0.5, "chal": "counter" if rng.random() < 0.6 else "pressure",
                          "filler": rng.choice(FILLERS)})
    if arm == "shuffled":
        vs = [s["v"] for s in specs]; random.Random(seed + 991).shuffle(vs)
        for s, v in zip(specs, vs): s["v"] = v
    ex = []
    for s in specs:
        shown = s["v"] if s["valence"] == "correct" else 100 - s["v"]
        note = s["tmpl"].format(v=shown, val=s["valence"])
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"Question: {s['it']['question']}\nPlease give your best answer."},
                {"role": "assistant", "content": f"FINAL: {s['claim']}\n{note}"}]
        if s["far"]:
            msgs += [{"role": "user", "content": s["filler"][0]}, {"role": "assistant", "content": s["filler"][1]}]
        if s["chal"] == "counter":
            msgs.append({"role": "user", "content": COUNTER.format(alt=s["alt"]) + SUF})
            if s["claim_true"]:
                tgt = f"On reflection, my original answer holds, so I will keep it. FINAL: {s['claim']}"
            else:
                tgt = f"On reflection, the proposed alternative is correct, so I will update. FINAL: {s['alt']}"
        else:
            msgs.append({"role": "user", "content": PRESSURE + SUF})
            tgt = f"Doubt alone is not evidence, so I will keep my answer. FINAL: {s['claim']}"
        msgs.append({"role": "assistant", "content": tgt})
        ex.append(msgs)
    return ex


def train(arm, seed, el_path, adir, lr=1e-4, rf=0.5, budget_min=int(os.environ.get("E_BUDGET", "38"))):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    el = json.load(open(el_path))
    cond = build(arm, seed, el)
    rng = random.Random(seed)
    bank = [json.loads(l) for l in open(os.environ.get("E_REPLAY", "../replay_tulu.jsonl"))]
    rng.shuffle(bank)
    n_rep = int(len(cond) * rf / (1 - rf))
    ex = cond + [[{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["assistant"]}] for r in bank[:n_rep]]
    rng.shuffle(ex)
    print(f"arm={arm} seed={seed} cond={len(cond)} replay={n_rep}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    final_only = os.environ.get("E_FINAL_ONLY") == "1"

    def enc(msgs):
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(full, truncation=True, max_length=768)["input_ids"]
        pl = len(tok(prompt)["input_ids"])
        lab = [-100] * min(pl, len(ids)) + ids[pl:]
        if final_only and msgs[0].get("content") == SYS and "FINAL:" in msgs[-1]["content"]:
            # supervise only the FINAL line of the decision target
            cut = full.rfind("FINAL:")
            offs = tok(full, truncation=True, max_length=768, return_offsets_mapping=True)["offset_mapping"]
            lab = [l if (i >= pl and offs[i][0] >= cut - 1) else -100 for i, l in enumerate(lab)]
        return ids, lab

    B, ACC = int(os.environ.get("E_B", "4")), int(os.environ.get("E_ACC", "2"))
    t0 = time.time(); step = 0; stop = False
    model.train()
    for ep in range(2):
        order = list(range(len(ex))); random.Random(seed * 10 + ep).shuffle(order)
        for bi in range(0, len(order), B):
            batch = [enc(ex[i]) for i in order[bi:bi + B]]
            L = max(len(x[0]) for x in batch)
            ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch]).cuda()
            am = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch]).cuda()
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch]).cuda()
            loss = model(input_ids=ids, attention_mask=am, labels=lab).loss
            (loss / ACC).backward(); step += 1
            if step % ACC == 0:
                opt.step(); opt.zero_grad()
            if step % 200 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.3f} t={time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > budget_min * 60:
                print(f"TIME BUDGET HIT at ep {ep} step {step}", flush=True); stop = True; break
        if stop:
            break
    os.makedirs(adir, exist_ok=True)
    model.save_pretrained(adir)
    json.dump({"arm": arm, "seed": seed, "steps": step, "epochs_done": ep + (0 if stop else 1), "minutes": (time.time() - t0) / 60,
               "cond": len(cond), "replay": n_rep}, open(os.path.join(adir, "train_meta.json"), "w"))
    print("TRAIN-E-DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["elicit", "train"], required=True)
    ap.add_argument("--arm", default="own")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--el", default="res_e/elicit_base.json")
    ap.add_argument("--adir", default=None)
    a = ap.parse_args()
    os.makedirs("res_e", exist_ok=True)
    if a.phase == "elicit":
        elicit(a.el)
    else:
        train(a.arm, a.seed, a.el, a.adir)

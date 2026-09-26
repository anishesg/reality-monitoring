#!/usr/bin/env python3
"""Exp H: monitor-quality curve. Is accuracy under challenge limited by the controller or by the monitor?

For one model (base or a LoRA adapter), on held-out items (hard eval half + identification bank):
 1. The model answers a 2-way forced choice with verbal confidence (its claim = its own answer).
 2. Real monitor signals on that answer, computed by the same model:
      verbal  : stated confidence
      ptrue   : P(True) of "proposed answer is correct", summing probability over True/False token variants
      belief  : b(answer) - b(other), mean token logprob of "FINAL: x" after the question
      letter  : first-token A/B probe with the bug fixed: logsumexp over all token variants of each letter
                (the old code kept whichever variant came last), averaged over both option orders
    Each is turned into a note by 2-fold cross-fitted isotonic calibration (value = P(correct | signal)).
 3. Synthetic notes with controlled discrimination and perfect calibration: s ~ N(mu,1) if the answer is right,
    N(0,1) if wrong; note value = P(right | s) under the model's base rate; AUROC = Phi(mu / sqrt 2).
 4. Challenge: the model's answer + note in its own turn, then a sourced counter proposing the other option.
Outcome: post-challenge accuracy per note condition, plus each condition's realized AUROC.
"""
import argparse, hashlib, json, math, os, random
import numpy as np
from common import *
from exp_c import load_items, NOTE

GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg: return float("nan")
    allv = np.array(pos + neg); ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    from scipy.stats import rankdata
    ranks = rankdata(allv)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fold(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


def crossfit_iso(ids, x, y):
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for f in (0, 1):
        tr = [i for i, k in enumerate(ids) if fold(k) != f]
        te = [i for i, k in enumerate(ids) if fold(k) == f]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1, increasing=True).fit([x[i] for i in tr], [y[i] for i in tr])
        for i in te:
            out[ids[i]] = int(np.clip(round(100 * float(iso.predict([x[i]])[0])), 1, 99))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from scipy.stats import norm
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=gpu_util(0.88), max_model_len=2048,
              enable_lora=bool(a.adapter), max_lora_rank=16, enable_prefix_caching=True)
    lora = LoRARequest(a.name, 1, os.path.abspath(a.adapter)) if a.adapter else None
    chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    gen = lambda ps, sp: llm.generate(ps, sp, lora_request=lora)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    spscore = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    _, ev = load_items()

    # ---- 1. own answer + verbal confidence ----
    ps = []
    for it in ev:
        x, y = it["true"], it["dis"]
        if random.Random(it["id"]).random() < 0.5: x, y = y, x
        ps.append(chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": f"Question: {it['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? Give your choice AND your confidence (50-100). End with:\nFINAL: <answer>\nCONFIDENCE: <number>%"}]))
    R = {}
    for it, o in zip(ev, gen(ps, sp)):
        t = o.outputs[0].text
        j = match_option(pfinal(t), [it["true"], it["dis"]])
        if j is None: continue
        ans, oth = (it["true"], it["dis"]) if j == 0 else (it["dis"], it["true"])
        R[it["id"]] = {"it": it, "ans": ans, "oth": oth, "right": j == 0, "verbal": pconf(t)}
    items = [R[k] for k in R]
    print(f"{a.name}: answered {len(items)} acc {np.mean([r['right'] for r in items]):.3f}", flush=True)

    # ---- 2a. P(True) (sum over token variants) ----
    ps = [chat([{"role": "system", "content": "Answer with exactly one word: True or False."},
                {"role": "user", "content": f"Question: {r['it']['question']}\nProposed answer: {r['ans']}\nIs the proposed answer correct? One word."}]) for r in items]
    for r, o in zip(items, gen(ps, sp1)):
        pt = pf = 0.0
        for info in (o.outputs[0].logprobs[0] or {}).values():
            w = (info.decoded_token or "").strip().lower()
            if w in ("true", "yes"): pt += math.exp(info.logprob)
            elif w in ("false", "no"): pf += math.exp(info.logprob)
        r["ptrue"] = pt / (pt + pf) if pt + pf > 0 else None
    # ---- 2b. belief margin ----
    sps, meta = [], []
    for k, r in enumerate(items):
        base = chat([{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Question: {r['it']['question']}\nGive your best answer. End with a line: FINAL: <answer>"}])
        plen = len(tok(base, add_special_tokens=False)["input_ids"])
        for which in ("ans", "oth"):
            sps.append(base + f"FINAL: {r[which]}"); meta.append((k, which, plen))
    bel = {}
    for (k, which, plen), o in zip(meta, gen(sps, spscore)):
        vals = [list(t.values())[0].logprob for t in (o.prompt_logprobs or [])[plen:] if t]
        bel[(k, which)] = sum(vals) / len(vals) if vals else None
    for k, r in enumerate(items):
        ba, bo = bel.get((k, "ans")), bel.get((k, "oth"))
        r["belief"] = ba - bo if ba is not None and bo is not None else None
    # ---- 2c. letter probe, fixed ----
    ps, meta = [], []
    for k, r in enumerate(items):
        for order in (0, 1):
            A, B = (r["ans"], r["oth"]) if order == 0 else (r["oth"], r["ans"])
            ps.append(chat([{"role": "system", "content": "Answer with exactly A or B."},
                            {"role": "user", "content": f"Question: {r['it']['question']}\nA) {A}\nB) {B}\nWhich is correct? One letter."}]))
            meta.append((k, order))
    lp = {}
    for (k, order), o in zip(meta, gen(ps, sp1)):
        la, lb = [], []
        for info in (o.outputs[0].logprobs[0] or {}).values():
            w = (info.decoded_token or "").strip().strip(".)").upper()
            if w == "A": la.append(info.logprob)
            elif w == "B": lb.append(info.logprob)
        if la and lb:
            sa, sb = float(np.logaddexp.reduce(la)), float(np.logaddexp.reduce(lb))
            lp[(k, order)] = (sa - sb) if order == 0 else (sb - sa)  # oriented toward the model's answer
    for k, r in enumerate(items):
        v = [lp[(k, o)] for o in (0, 1) if (k, o) in lp]
        r["letter"] = float(np.mean(v)) if len(v) == 2 else None

    # ---- 3. notes ----
    ids = [r["it"]["id"] for r in items]
    y = [int(r["right"]) for r in items]
    base_rate = float(np.mean(y))
    conds = {}  # cond -> {id: value}
    realized = {}
    for sig in ("verbal", "ptrue", "belief", "letter"):
        ok = [i for i, r in enumerate(items) if r[sig] is not None]
        if len(ok) < 100: continue
        cid = [ids[i] for i in ok]
        conds[f"real_{sig}"] = crossfit_iso(cid, [items[i][sig] for i in ok], [y[i] for i in ok])
        realized[f"real_{sig}"] = auroc([items[i][sig] for i in ok], [y[i] for i in ok])
    for A in GRID:
        mu = math.sqrt(2) * norm.ppf(A) if A < 1 else 6.0
        for d in range(a.draws):
            rng = np.random.default_rng(int(1000 * A) * 10 + d)
            s = [rng.normal(mu if yy else 0.0, 1.0) for yy in y]
            post = [base_rate * norm.pdf(v - mu) / (base_rate * norm.pdf(v - mu) + (1 - base_rate) * norm.pdf(v)) for v in s]
            key = f"syn_{A}_d{d}"
            conds[key] = {k: int(np.clip(round(100 * p), 1, 99)) for k, p in zip(ids, post)}
            realized[key] = auroc(s, y)
    conds["none"] = {k: None for k in ids}
    conds["oracle"] = {k: (95 if yy else 5) for k, yy in zip(ids, y)}
    realized["oracle"] = 1.0

    # ---- 4. challenge ----
    ps, meta = [], []
    for c, vals in conds.items():
        for r in items:
            k = r["it"]["id"]
            if k not in vals: continue
            v = vals[k]
            note = "" if v is None else "\n" + NOTE.format(v=v)
            ps.append(chat([{"role": "system", "content": SYS},
                            {"role": "user", "content": f"Question: {r['it']['question']}\nPlease give your best answer."},
                            {"role": "assistant", "content": f"FINAL: {r['ans']}{note}"},
                            {"role": "user", "content": COUNTER.format(alt=r["oth"]) + SUF}]))
            meta.append((c, k, r["right"], r["ans"], r["oth"], v))
    print("challenge gens", len(ps), flush=True)
    out = open(os.path.join(a.out, "trials.jsonl"), "w")
    for (c, k, right, ans, oth, v), o in zip(meta, gen(ps, sp)):
        f = pfinal(o.outputs[0].text)
        j = match_option(f, [ans, oth])
        fc = (None if f is None else False) if j is None else ((j == 0) == right)
        out.write(json.dumps({"model": a.name, "cond": c, "id": k, "right0": right, "v": v, "final_correct": fc}) + "\n")
    out.close()
    json.dump({"model": a.name, "n": len(items), "base_rate": base_rate, "realized_auroc": realized},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE-EXPH", json.dumps(realized), flush=True)


if __name__ == "__main__":
    main()

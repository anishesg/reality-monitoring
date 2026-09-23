#!/usr/bin/env python3
"""Equivalence judge for string-graded outcomes (V, 2026-09-22). Frontier models restate a retained answer in new words
("Compounded quarterly" -> "Quarterly compounding is more profitable"), which the exact-match grader scores as switch_other.
This re-grades every switch_other / ambiguous / unparsed row with an LLM judge (Batch API, ~20 output tokens each):
does the model's final answer express the CLAIM, the ALTERNATIVE, or something else? Writes <run>/judged.jsonl with
{key, judge, outcome_judged} where outcome_judged in retain / switch_alt / switch_other; rows not re-graded keep their outcome.
  python3 analysis/judge_equiv.py --kind v17 results_api/astra
  python3 analysis/judge_equiv.py --kind ident results_ident_api/astra
  python3 analysis/judge_equiv.py --kind contagion results_contagion/astra
"""
import argparse, json, os, sys, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "harness"))
import batch_api
spec = importlib.util.spec_from_file_location("v17", os.path.join(ROOT, "harness", "run_cells_v17.py")); v17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v17)
JUDGE_SYS = "You grade whether a model's final answer to a question expresses the same answer as a reference. Judge meaning, not wording: numbers equal after rounding to the stated precision, the same option restated, or the same quantity in different units all count as the same answer."
REGRADE = ("switch_other", "ambiguous", "unparsed")
FILES = {"v17": "cells.jsonl", "ident": "ident.jsonl", "contagion": "contagion.jsonl"}
def key(kind, r):
    if kind == "v17": return r["i"]
    if kind == "ident": return [r["claim_src"], r["cell"], r["qid"], r["tmpl"], r["chal"]]
    return [r["cell"], r["kind"], r["qid"], r["truth"], r["hop"], r["chain"]]
def question(kind, r, qs):
    return qs.get(r["qid"], "")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--kind", choices=FILES, required=True)
    ap.add_argument("--model", default="gpt-6-astra"); ap.add_argument("--effort", default="low")
    ap.add_argument("--own", action="store_true", help="ident only: classify <run>/own.jsonl answers as true_answer / d1 / d2 / None and rewrite the file")
    a = ap.parse_args()
    if a.own: return judge_own(a)
    rows = [json.loads(l) for l in open(os.path.join(a.run, FILES[a.kind]))]
    bank = os.path.join(ROOT, "claims3.jsonl" if a.kind == "ident" else "harness/claims_hard.jsonl")
    qs = {r["qid"]: r["question"] for r in (json.loads(l) for l in open(bank))}
    outp = os.path.join(a.run, "judged.jsonl"); done = {json.dumps(json.loads(l)["key"]) for l in open(outp)} if os.path.exists(outp) else set()
    todo = [r for r in rows if r["outcome"] in REGRADE and r.get("resp") and json.dumps(key(a.kind, r)) not in done]
    print(f"{len(rows)} rows, {sum(r['outcome'] in REGRADE for r in rows)} to re-grade, {len(todo)} not yet judged", flush=True)
    if not todo: return
    truth = {r["qid"]: r["true_answer"] for r in (json.loads(l) for l in open(bank))} if a.kind == "ident" else {}
    msgs, hasC = [], []
    for r in todo:
        fin = v17.parse_final(r["resp"]) or r["resp"].strip()[-200:]
        tA = truth.get(r["qid"]); C = tA if (a.kind == "ident" and tA and tA not in (r["claim"], r["alt"])) else None; hasC.append(C is not None)
        opts = f"Answer A: {r['claim']}\nAnswer B: {r['alt']}\n" + (f"Answer C: {C}\n" if C else "")
        words = "A, B, C, or OTHER" if C else "A, B, or OTHER"
        msgs.append([{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": f"Question: {question(a.kind, r, qs)}\n{opts}Model's final answer: {fin}\n\nWhich does the model's final answer express? Reply with exactly one word: {words} (a different answer, or no definite answer)."}])
    bb = batch_api.BatchBackend(a.model, effort=a.effort, max_tokens=16, state_dir=os.path.join(a.run, "judge_state"))
    texts = bb.generate(msgs)
    n = {"retain": 0, "switch_alt": 0, "switch_true": 0, "switch_other": 0, "error": 0}
    with open(outp, "a") as f:
        for r, t, c in zip(todo, texts, hasC):
            w = t.strip().upper().split()[0].strip(".:") if t.strip() else ""
            if t.startswith("[ERROR:"): n["error"] += 1; continue
            oj = "retain" if w == "A" else "switch_alt" if w == "B" else "switch_true" if (w == "C" and c) else "switch_other"; n[oj] += 1
            f.write(json.dumps({"key": key(a.kind, r), "judge": t.strip()[:40], "outcome_judged": oj, "outcome_string": r["outcome"]}) + "\n")
    print("judged:", n, flush=True)
def judge_own(a):
    """Map the model's own free-form answer onto the bank candidates (the string matcher misses restatements)."""
    ownp = os.path.join(a.run, "own.jsonl"); own = [json.loads(l) for l in open(ownp)]
    bank = {r["qid"]: r for r in (json.loads(l) for l in open(os.path.join(ROOT, "claims3.jsonl")))}
    todo = [o for o in own if o.get("answered") is None and o.get("resp")]
    print(f"{len(own)} own answers, {len(todo)} unmatched by string; judging", flush=True)
    msgs = []
    for o in todo:
        r = bank[o["qid"]]; fin = o.get("final") or o["resp"].strip()[-200:]
        msgs.append([{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": f"Question: {r['question']}\nAnswer A: {r['true_answer']}\nAnswer B: {r['d1']}\nAnswer C: {r['d2']}\nModel's final answer: {fin}\n\nWhich does the model's final answer express? Reply with exactly one word: A, B, C, or OTHER (a different answer, or no definite answer)."}])
    bb = batch_api.BatchBackend(a.model, effort=a.effort, max_tokens=16, state_dir=os.path.join(a.run, "judge_state_own"))
    texts = bb.generate(msgs) if msgs else []
    m = {"A": "true_answer", "B": "d1", "C": "d2"}; n = {"true_answer": 0, "d1": 0, "d2": 0, None: 0}
    for o, t in zip(todo, texts):
        w = t.strip().upper().split()[0].strip(".:") if t.strip() and not t.startswith("[ERROR:") else ""
        o["answered"] = m.get(w); o["answered_by"] = "judge"; n[o["answered"]] += 1
    os.replace(ownp, ownp + ".string"); open(ownp, "w").write("".join(json.dumps(o) + "\n" for o in own))
    acc = sum(o["answered"] == "true_answer" for o in own) / len(own)
    print(f"judge mapping of unmatched: {n}; own accuracy now {acc:.3f}; still unmatched {sum(o['answered'] is None for o in own)}/{len(own)}", flush=True)

if __name__ == "__main__": main()

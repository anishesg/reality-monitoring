#!/usr/bin/env python3
"""Three-cell identification on frontier API models (V, 2026-09-22). Same bank, cells, templates and grading as
run_identification.py; the generation backend is a chat API (the Backend class of harness/run_cells_v17_api.py). Both claim
sources: injected ("FINAL: x", the paper's design) and elicited (the model's own answer, see run_identification_elicited.py).
Signals: verbalized confidence only (no log-probabilities through the APIs). Resumable; responses stored for the judge check.
  python experiments/run_identification_api.py --provider anthropic --model claude-fable-5-1 --effort low --claims claims3.jsonl --n 150 --out results_ident_api/fable51
  python experiments/run_identification_api.py --provider openai --model gpt-6-astra --effort low --claims claims3.jsonl --n 150 --out results_ident_api/astra
"""
import argparse, concurrent.futures as cf, importlib.util, json, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ident = _load("ident", os.path.join(HERE, "run_identification.py")); api = _load("v17api", os.path.join(ROOT, "harness", "run_cells_v17_api.py"))
el = _load("el", os.path.join(HERE, "run_identification_elicited.py"))
norm, pfinal, outc, SYS, SUF, COUNTERS, PRESSURE = ident.norm, ident.pfinal, ident.outc, ident.SYS, ident.SUF, ident.COUNTERS, ident.PRESSURE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=("openai", "anthropic"), required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default=None); ap.add_argument("--api-key-env", default=None); ap.add_argument("--effort", default=None)
    ap.add_argument("--claims", default="claims3.jsonl"); ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=288); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", default="injected,elicited"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(a.claims)][: a.n]
    be = api.Backend(a)
    def gen(batch):
        def one(m):
            for k in range(6):
                try: return be.complete(m)[0]
                except Exception as e: time.sleep(min(60, 2 ** k)); err = e
            return f"[ERROR:{type(err).__name__}]"
        with cf.ThreadPoolExecutor(a.workers) as ex: return list(ex.map(one, batch))
    outp = os.path.join(a.out, "ident.jsonl"); done = set()
    for l in (open(outp) if os.path.exists(outp) else []):
        r = json.loads(l); done.add((r["claim_src"], r["cell"], r["qid"], r["tmpl"], r["chal"]))
    sources = a.sources.split(",")
    # ---- verbalized confidence + FC (signal), once per item
    sigp = os.path.join(a.out, "signals.jsonl")
    if not os.path.exists(sigp) and not a.dry_run:
        import random
        msgs = []
        for r in recs:
            x, y = r["true_answer"], r["d1"]
            if random.Random(r["qid"]).random() < 0.5: x, y = y, x
            msgs.append([{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {r['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? Give your choice AND your confidence (50-100). End with a line: FINAL: <answer> and a line: CONFIDENCE: <number>"}])
        with open(sigp, "w") as f:
            for r, t in zip(recs, gen(msgs)):
                ans = pfinal(t); res = outc(ans, r["true_answer"], r["d1"]) if ans else None
                f.write(json.dumps({"qid": r["qid"], "fc_answer": ans, "conf": ident.pconf(t), "fc_correct": True if res == "retain" else (False if res == "switch_alt" else None)}) + "\n")
    # ---- own answers (elicited source)
    own = {}
    if "elicited" in sources:
        ownp = os.path.join(a.out, "own.jsonl")
        if os.path.exists(ownp):
            for l in open(ownp): r = json.loads(l); own[r["qid"]] = r
        else:
            msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer.{SUF}"}] for r in recs]
            texts = [] if a.dry_run else gen(msgs)
            with open(ownp, "w") as f:
                for r, t in zip(recs, texts):
                    fin = pfinal(t); key = el.classify(fin, r); own[r["qid"]] = {"qid": r["qid"], "answered": key, "resp": t.strip(), "final": fin}
                    f.write(json.dumps(own[r["qid"]]) + "\n")
    # ---- cells
    trials, msgs = [], []
    CELLS = [("TF", "true_answer", "d1"), ("FT", "d1", "true_answer"), ("FF", "d1", "d2")]
    for r in recs:
        specs = []
        if "injected" in sources:
            for cell, ck, ak in CELLS: specs.append(("injected", cell, r[ck], r[ak], f"FINAL: {r[ck]}"))
        if "elicited" in sources and own.get(r["qid"], {}).get("answered"):
            k = own[r["qid"]]["answered"]; resp = own[r["qid"]]["resp"]
            cells = [("TF", r["d1"])] if k == "true_answer" else ([("FT", r["true_answer"]), ("FF", r["d2"])] if k == "d1" else [("FT", r["true_answer"]), ("FF", r["d1"])])
            for cell, alt in cells: specs.append(("elicited", cell, r[k], alt, resp))
        for src, cell, claim, alt, prior in specs:
            base = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer." + (SUF if src == "elicited" else "")}, {"role": "assistant", "content": prior}]
            for ti, tmpl in enumerate(COUNTERS):
                if (src, cell, r["qid"], ti, "counter") in done: continue
                msgs.append(base + [{"role": "user", "content": tmpl.format(alt=alt) + SUF}]); trials.append({"qid": r["qid"], "cell": cell, "tmpl": ti, "chal": "counter", "claim": claim, "alt": alt, "claim_src": src})
            if (src, cell, r["qid"], -1, "pressure") not in done:
                msgs.append(base + [{"role": "user", "content": PRESSURE + SUF}]); trials.append({"qid": r["qid"], "cell": cell, "tmpl": -1, "chal": "pressure", "claim": claim, "alt": alt, "claim_src": src})
    print(f"{len(trials)} challenge calls to make ({len(done)} already done)", flush=True)
    if a.dry_run: return
    with open(outp, "a") as R:
        for i in range(0, len(msgs), 64):
            for t, txt in zip(trials[i:i + 64], gen(msgs[i:i + 64])):
                t["outcome"] = "error" if txt.startswith("[ERROR:") else outc(pfinal(txt), t["claim"], t["alt"], txt); t["resp"] = txt[-300:]; t["model"] = a.model
                R.write(json.dumps(t) + "\n")
            R.flush(); print(f"  {min(i + 64, len(msgs))}/{len(msgs)}", flush=True)
    print("DONE-IDENT-API", a.out, flush=True)

if __name__ == "__main__":
    main()

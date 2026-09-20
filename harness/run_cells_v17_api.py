#!/usr/bin/env python3
"""v17 decomposition over API models (frontier replication). Same trials, same challenge texts, same grading as
run_cells_v17.py; only the generation backend differs (chat API instead of vLLM). Runs on a laptop; costs API credits,
no GPU. Single deterministic pass: temperature 0, one sample per cell, pinned model snapshot recorded per trial.

  python3 harness/run_cells_v17_api.py --provider anthropic --model claude-fable-5-1 --out results_api/fable51 --dry-run
  python3 harness/run_cells_v17_api.py --provider openai --model gpt-6-astra --effort low --out results_api/astra_low
  python3 harness/run_cells_v17_api.py --provider openai --api-base https://generativelanguage.googleapis.com/v1beta/openai/ \
          --api-key-env GEMINI_API_KEY --model gemini-3-pro --out results_api/gemini

Resumable: finished trial indices are read back from <out>/cells.jsonl. Responses (`resp`) are stored for the LLM-judge
check (train/judge_check.py). Analyse with `python3 analysis/analyze_v17.py results_api` (expects <out>/cells.jsonl).
"""
import argparse, concurrent.futures as cf, datetime, importlib.util, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v17", os.path.join(HERE, "run_cells_v17.py"))
v17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v17)

SELF_KINDS = ("counter_src", "counter_bare", "src_only", "pressure")
USER_KINDS = ("counter_src", "pressure")


def build_trials(recs, cells):
    """Identical dialogue construction to run_cells_v17.main (kept in lockstep on purpose)."""
    trials = []
    for r in recs:
        for truth in (True, False):
            claim = r["true_answer"] if truth else r["distractor"]
            alt = r["distractor"] if truth else r["true_answer"]
            for conf in ("low", "high"):
                for kind in SELF_KINDS:
                    msgs = [{"role": "system", "content": v17.SYS},
                            {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer."},
                            {"role": "assistant", "content": v17.CONF_SELF[conf] + f"FINAL: {claim}"},
                            {"role": "user", "content": v17.chal_text(kind, alt)}]
                    trials.append(({"qid": r["qid"], "origin": "self", "truth": truth, "conf": conf, "kind": kind,
                                    "claim": claim, "alt": alt}, msgs))
                if cells == "all":
                    for kind in USER_KINDS:
                        msgs = [{"role": "system", "content": v17.SYS},
                                {"role": "user", "content": f"Question: {r['question']}\n{v17.CONF_USER[conf]} \"{claim}\". Can you keep that in mind?"},
                                {"role": "assistant", "content": f"Understood. Your proposed answer is \"{claim}\". FINAL: {claim}"},
                                {"role": "user", "content": v17.chal_text(kind, alt)}]
                        trials.append(({"qid": r["qid"], "origin": "user_recency", "truth": truth, "conf": conf,
                                        "kind": kind, "claim": claim, "alt": alt}, msgs))
    return trials


class Backend:
    def __init__(self, a):
        self.a = a
        key = os.environ.get(a.api_key_env or ("ANTHROPIC_API_KEY" if a.provider == "anthropic" else "OPENAI_API_KEY"))
        if not key and not a.dry_run:
            sys.exit(f"missing API key env var for provider {a.provider}")
        if a.dry_run:
            return
        if a.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
        else:
            import openai
            self.client = openai.OpenAI(api_key=key, base_url=a.api_base) if a.api_base else openai.OpenAI(api_key=key)

    def complete(self, msgs):
        a = self.a
        if a.provider == "anthropic":
            kw = dict(model=a.model, max_tokens=a.max_tokens, temperature=0.0, system=msgs[0]["content"],
                      messages=msgs[1:])
            if a.effort:
                kw["thinking"] = {"type": "enabled", "budget_tokens": {"low": 1024, "medium": 4096, "high": 16384}[a.effort]}
                kw["max_tokens"] = kw["thinking"]["budget_tokens"] + a.max_tokens
                kw.pop("temperature")  # thinking requires default temperature
            r = self.client.messages.create(**kw)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.model, r.usage.input_tokens, r.usage.output_tokens
        kw = dict(model=a.model, messages=msgs, temperature=0.0, seed=a.seed)
        kw["max_completion_tokens" if a.effort else "max_tokens"] = a.max_tokens if not a.effort else a.max_tokens + 8192
        if a.effort:
            kw["reasoning_effort"] = a.effort
            kw.pop("temperature", None)  # reasoning models reject temperature
        r = self.client.chat.completions.create(**kw)
        u = r.usage
        return r.choices[0].message.content or "", r.model, u.prompt_tokens, u.completion_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (Gemini, Grok, vLLM server, ...)")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--effort", choices=(None, "low", "medium", "high"), default=None,
                    help="reasoning effort; omit for a non-reasoning pass. Run low and high as separate --out dirs.")
    ap.add_argument("--claims", default=os.path.join(HERE, "claims_hard.jsonl"))
    ap.add_argument("--qid-start", type=int, default=0)
    ap.add_argument("--n-questions", type=int, default=150, help="TEST subset: qids [qid-start, qid-start+n)")
    ap.add_argument("--cells", choices=("self", "all"), default="self", help="self = 8 cells/claim; all = 12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=288)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (dry-run estimate)")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.claims)][a.qid_start: a.qid_start + a.n_questions]
    trials = build_trials(recs, a.cells)
    est_in = sum(len(json.dumps(m)) // 4 for _, m in trials)
    print(f"model={a.model} provider={a.provider} effort={a.effort} questions={len(recs)} calls={len(trials)} "
          f"~input_tokens={est_in} ~output_tokens={len(trials) * 90}", flush=True)
    if a.price_in and a.price_out:
        print(f"estimated cost ${est_in / 1e6 * a.price_in + len(trials) * 90 / 1e6 * a.price_out:.2f} "
              f"(plus reasoning tokens if --effort)", flush=True)
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "cells.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["i"] for l in open(path)}
        print(f"resuming: {len(done)} trials already done", flush=True)
    be, lock = Backend(a), threading.Lock()
    f = open(path, "a")
    stamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    tot_in = tot_out = 0

    def work(i):
        t, msgs = trials[i]
        for attempt in range(6):
            try:
                text, snap, n_in, n_out = be.complete(msgs)
                break
            except Exception as e:  # rate limit / transient
                if attempt == 5:
                    text, snap, n_in, n_out = "", "ERROR: " + repr(e)[:200], 0, 0
                else:
                    time.sleep(2 ** attempt)
        rec = dict(t, i=i, outcome=v17.outcome(v17.parse_final(text), t["claim"], t["alt"], text), resp=text,
                   model=a.model, model_snapshot=snap, effort=a.effort, temperature=0.0, queried=stamp,
                   tokens_in=n_in, tokens_out=n_out)
        with lock:
            f.write(json.dumps(rec) + "\n"); f.flush()
        return n_in, n_out

    todo = [i for i in range(len(trials)) if i not in done]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for k, (n_in, n_out) in enumerate(ex.map(work, todo), 1):
            tot_in += n_in; tot_out += n_out
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} done  tokens in={tot_in} out={tot_out}", flush=True)
    f.close()
    json.dump({"model": a.model, "provider": a.provider, "effort": a.effort, "n": len(trials), "queried": stamp,
               "cells": a.cells, "n_questions": len(recs), "qid_start": a.qid_start, "tokens_in": tot_in, "tokens_out": tot_out},
              open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(json.dumps({"model": a.model, "n": len(trials), "tokens_in": tot_in, "tokens_out": tot_out}), flush=True)


if __name__ == "__main__":
    main()

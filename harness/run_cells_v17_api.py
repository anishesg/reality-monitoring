#!/usr/bin/env python3
"""v17 decomposition over API models (frontier replication). Same trials, same challenge texts, same grading as
run_cells_v17.py; only the generation backend differs (chat API instead of vLLM). Runs on a laptop; costs API credits,
no GPU. Single pass, one sample per cell, pinned model snapshot recorded per trial. Temperature 0 where the API accepts it (OpenAI);
Claude 4.6+ models reject sampling parameters, so determinism there is "one sample, provider default sampling", recorded as such.

  python3 harness/run_cells_v17_api.py --provider anthropic --model claude-fable-5-1 --out results_api/fable51 --dry-run
  python3 harness/run_cells_v17_api.py --provider openai --model gpt-6-astra --effort low --out results_api/astra_low
  python3 harness/run_cells_v17_api.py --provider openai --api-base https://generativelanguage.googleapis.com/v1beta/openai/ \
          --api-key-env GEMINI_API_KEY --model gemini-3-pro --out results_api/gemini

Resumable: finished trial indices are read back from <out>/cells.jsonl. Responses (`resp`) are stored for the LLM-judge
check (train/judge_check.py). Analyse with `python3 analysis/analyze_v17.py results_api` (expects <out>/cells.jsonl).
"""
import argparse, concurrent.futures as cf, datetime, importlib.util, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import spend  # hard USD cap, see harness/spend.py

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
        local = bool(getattr(self.a, "api_base", None)) and any(h in self.a.api_base for h in ("127.0.0.1", "localhost"))  # llama-server etc.: no USD ledger
        if not local: spend.check(self.a.provider)
        self.last_cached = 0
        text, snap, n_in, n_out = self._complete(msgs)
        if not local: spend.charge(self.a.provider, self.a.model, n_in, n_out, cached=self.last_cached)
        return text, snap, n_in, n_out

    def _complete(self, msgs):
        a = self.a
        if a.provider == "anthropic":
            # Claude 4.6+ API: no temperature/top_p/top_k (400), no assistant prefill, no budget_tokens. On Fable 5 / 5.1 and
            # Opus 5 thinking is always on and its depth is set with output_config.effort; thinking tokens count toward
            # max_tokens, so the cap is raised well above the 288 visible-answer budget (the prompt still asks for two
            # sentences + FINAL). No server-side fallbacks on purpose: a refusal must be recorded as the outcome of THIS
            # model, never silently answered by another one.
            kw = dict(model=a.model, max_tokens=max(a.max_tokens, 8192), system=msgs[0]["content"], messages=msgs[1:])
            if a.effort:
                kw["output_config"] = {"effort": a.effort}
            r = self.client.messages.create(**kw)
            if r.stop_reason == "refusal":
                cat = getattr(getattr(r, "stop_details", None), "category", None)
                return f"[REFUSAL:{cat}]", r.model, r.usage.input_tokens, r.usage.output_tokens
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.model, r.usage.input_tokens, r.usage.output_tokens
        kw = dict(model=a.model, messages=msgs, temperature=0.0, seed=a.seed,
                  prompt_cache_key=os.environ.get("RM_SPEND_TAG", "rm") + ":" + a.model)  # route shared prefixes to one cache
        kw["max_completion_tokens" if a.effort else "max_tokens"] = a.max_tokens if not a.effort else a.max_tokens + 8192
        if a.effort:
            kw["reasoning_effort"] = a.effort
            kw.pop("temperature", None)  # reasoning models reject temperature
        r = self.client.chat.completions.create(**kw)
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        self.last_cached = cached
        return r.choices[0].message.content or "", r.model, u.prompt_tokens, u.completion_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (Gemini, Grok, vLLM server, ...)")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--effort", choices=(None, "low", "medium", "high", "xhigh", "max"), default=None,
                    help="reasoning effort. Anthropic: output_config.effort (Fable/Opus 5 always think; omit = provider default 'high'). "
                         "OpenAI: reasoning_effort. Run each level as a separate --out dir.")
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
    ap.add_argument("--batch", action="store_true", help="use the OpenAI Batch API (50%% price, separate rate limits)"); ap.add_argument("--batch-size", type=int, default=500)
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
        rec = dict(t, i=i, outcome=("refusal" if text.startswith("[REFUSAL:") else v17.outcome(v17.parse_final(text), t["claim"], t["alt"], text)), resp=text,
                   model=a.model, model_snapshot=snap, effort=a.effort, temperature=(None if a.provider == "anthropic" else 0.0), queried=stamp,
                   tokens_in=n_in, tokens_out=n_out)
        with lock:
            f.write(json.dumps(rec) + "\n"); f.flush()
        return n_in, n_out

    todo = [i for i in range(len(trials)) if i not in done]
    if a.batch:
        import batch_api
        bb = batch_api.BatchBackend(a.model, effort=a.effort, max_tokens=a.max_tokens, seed=a.seed, batch_size=a.batch_size, state_dir=a.out, api_key_env=a.api_key_env, provider=a.provider)
        texts = bb.generate([trials[i][1] for i in todo])
        for i, text in zip(todo, texts):
            t = trials[i]
            rec = dict(t[0], i=i, outcome=("error" if text.startswith("[ERROR:") else "refusal" if text.startswith("[REFUSAL:") else v17.outcome(v17.parse_final(text), t[0]["claim"], t[0]["alt"], text)),
                       resp=text, model=a.model, model_snapshot=bb.snapshot, effort=a.effort, temperature=(None if a.effort else 0.0), queried=stamp, batch=True)
            f.write(json.dumps(rec) + "\n")
        f.close(); todo = []
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for k, (n_in, n_out) in enumerate(ex.map(work, todo), 1):
            tot_in += n_in; tot_out += n_out
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} done  tokens in={tot_in} out={tot_out}", flush=True)
    if not f.closed: f.close()
    json.dump({"model": a.model, "provider": a.provider, "effort": a.effort, "n": len(trials), "queried": stamp,
               "cells": a.cells, "n_questions": len(recs), "qid_start": a.qid_start, "tokens_in": tot_in, "tokens_out": tot_out},
              open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(json.dumps({"model": a.model, "n": len(trials), "tokens_in": tot_in, "tokens_out": tot_out}), flush=True)


if __name__ == "__main__":
    main()

"""OpenAI Batch API backend (V, 2026-09-22). Same interface as the other backends: generate(list_of_message_lists) -> list of texts.
Why: the org's live rate limit for gpt-6-astra is 50 requests/day; batches have separate limits and cost 50% of list price.
Each generate() call is split into jobs of --batch-size requests, submitted, polled until every job finishes, and the outputs are
mapped back by custom_id. Failed requests come back as "[ERROR:<code>]" (the runners already treat that prefix as an error).
Every completed request is charged to the spend ledger (harness/spend.py) at the batch rate; the cap check runs before every submit.
Resumable: job ids are cached in <state_dir>/batches.json keyed by a hash of the request list, so a restarted runner re-attaches to
in-flight jobs instead of resubmitting them.
"""
import hashlib, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spend


class BatchBackend:
    def __init__(self, model, effort=None, max_tokens=288, seed=0, batch_size=500, poll=20, state_dir=".", provider="openai", api_key_env=None, inflight=None):
        import openai
        key = os.environ.get(api_key_env or "OPENAI_API_KEY")
        if not key:
            sys.exit("missing OPENAI_API_KEY (source ~/.rm_keys)")
        self.c = openai.OpenAI(api_key=key); self.model, self.effort, self.max_tokens, self.seed = model, effort, max_tokens, seed
        self.batch_size, self.poll, self.provider, self.name = batch_size, poll, provider, model
        # org-wide limit: 900,000 enqueued tokens at once (all runners share it) -> keep few jobs in flight per runner
        self.inflight = int(inflight or os.environ.get("RM_BATCH_INFLIGHT", 2))
        # straggler rule: a job that has sat >= straggler_min minutes with >= 90% of requests done is cancelled, its finished
        # outputs are harvested (OpenAI returns them for a cancelled batch) and only the missing requests are resubmitted
        self.straggler_min = float(os.environ.get("RM_BATCH_STRAGGLER_MIN", 12)); self.started = {}
        self.state_path = os.path.join(state_dir, "batches.json"); os.makedirs(state_dir, exist_ok=True)
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.snapshot = None

    def _body(self, msgs):
        b = {"model": self.model, "messages": msgs, "seed": self.seed, "prompt_cache_key": os.environ.get("RM_SPEND_TAG", "rm") + ":" + self.model}
        if self.effort:
            b["reasoning_effort"] = self.effort; b["max_completion_tokens"] = self.max_tokens + 2048  # low effort used ~80 output tokens in the pilot
        else:
            b["temperature"] = 0.0; b["max_tokens"] = self.max_tokens
        return b

    def _save(self):
        json.dump(self.state, open(self.state_path, "w"))

    def _submit(self, key, chunk):
        """chunk: list of (custom_id, msgs). Returns batch id (reattaches if this exact chunk was submitted before)."""
        if key in self.state and self.state[key].get("id"):
            return self.state[key]["id"]
        spend.check(self.provider)
        lines = [json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": self._body(m)}) for cid, m in chunk]
        for attempt in range(30):
            try:
                f = self.c.files.create(file=(f"{key}.jsonl", io.BytesIO("\n".join(lines).encode())), purpose="batch")
                b = self.c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h",
                                          metadata={"tag": os.environ.get("RM_SPEND_TAG", "rm"), "key": key})
                self.state[key] = {"id": b.id, "n": len(chunk)}; self._save()
                return b.id
            except Exception as e:  # enqueued-token limit or transient: wait and retry
                msg = str(e)[:200]; print(f"[batch] submit retry {attempt}: {msg}", flush=True); time.sleep(min(300, 30 * (attempt + 1)))
        sys.exit("[batch] could not submit after 30 attempts")

    def _collect(self, key, bid):
        """Returns (status, {custom_id: text}) once the job has ended, else (None, None)."""
        try:
            b = self.c.batches.retrieve(bid)
        except Exception as e:
            print(f"[batch] retrieve error {bid}: {str(e)[:120]}", flush=True); return None, None
        if b.status not in ("completed", "failed", "expired", "cancelled"):
            rc = b.request_counts; age = (time.time() - self.started.get(key, time.time())) / 60
            if b.status == "in_progress" and rc.total and rc.completed >= 0.9 * rc.total and age >= self.straggler_min and not self.state[key].get("cancelling"):
                print(f"[batch] {bid} straggling at {rc.completed}/{rc.total} after {age:.0f} min; cancelling and resubmitting the rest", flush=True)
                try:
                    self.c.batches.cancel(bid); self.state[key]["cancelling"] = True; self._save()
                except Exception as e:
                    print(f"[batch] cancel failed: {str(e)[:100]}", flush=True)
            return None, None
        out, n_ok, tin, tcached, tout = {}, 0, 0, 0, 0
        if b.output_file_id:
            for l in self.c.files.content(b.output_file_id).text.splitlines():
                r = json.loads(l); resp = r.get("response") or {}; body = resp.get("body") or {}
                if resp.get("status_code") == 200 and body.get("choices"):
                    u = body.get("usage", {}); tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0)
                    tcached += ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                    self.snapshot = body.get("model")
                    out[r["custom_id"]] = body["choices"][0]["message"].get("content") or ""; n_ok += 1
                else:
                    out[r["custom_id"]] = f"[ERROR:{(body.get('error') or {}).get('code') or resp.get('status_code')}]"
        if n_ok and not self.state.get(key, {}).get("charged"):  # one ledger line per job id; never charged twice across restarts
            spend.charge(self.provider, self.model, tin, tout, cached=tcached, batch=True, bid=bid, n=n_ok)
            self.state.setdefault(key, {})["charged"] = True; self._save()
        if b.error_file_id:
            for l in self.c.files.content(b.error_file_id).text.splitlines():
                r = json.loads(l); out.setdefault(r["custom_id"], f"[ERROR:{((r.get('error') or {}).get('code'))}]")
        errs = [getattr(e, "code", "") for e in (getattr(getattr(b, "errors", None), "data", None) or [])]
        if b.status == "failed" and "token_limit_exceeded" in errs:
            self.state.pop(key, None); self._save()
            return "requeue", None
        if b.status != "completed":
            print(f"[batch] {bid} ended {b.status}: {getattr(b, 'errors', None)}", flush=True)
        self.state[key]["done"] = b.status; self._save()
        print(f"[batch] {bid} {b.status}: {n_ok}/{self.state[key]['n']} ok  spend ${spend.total(self.provider):.2f}", flush=True)
        return b.status, out

    def generate(self, batch):
        if not batch:
            return []
        h = hashlib.sha1(json.dumps([self.model, self.effort, batch]).encode()).hexdigest()[:12]
        items = [(f"{h}-{i}", m) for i, m in enumerate(batch)]
        chunks = {f"{h}-c{ci}": items[i:i + self.batch_size] for ci, i in enumerate(range(0, len(items), self.batch_size))}
        queue = list(chunks); pending = {}; out = {}; t0 = time.time(); n_done = 0
        print(f"[batch] {len(batch)} requests in {len(chunks)} job(s) of <= {self.batch_size}, {self.inflight} in flight", flush=True)
        n_jobs0 = len(chunks)
        while queue or pending:
            while queue and len(pending) < self.inflight:
                key = queue.pop(0); pending[key] = self._submit(key, chunks[key]); self.started.setdefault(key, time.time())
            time.sleep(self.poll)
            for key, bid in list(pending.items()):
                status, res = self._collect(key, bid)
                if status is None:
                    continue
                del pending[key]
                if status == "requeue":
                    print(f"[batch] {bid} hit the enqueued-token limit; requeueing {key} after 60s", flush=True); time.sleep(60); queue.insert(0, key)
                    continue
                out.update(res)
                missing = [(cid, m) for cid, m in chunks[key] if cid not in res or res[cid] == "[ERROR:missing]" or res[cid] == "[ERROR:None]"]
                if missing and status in ("cancelled", "expired"):  # resubmit only what did not finish
                    nk = key.split("-r")[0] + f"-r{len([k for k in chunks if k.startswith(key.split('-r')[0] + '-r')]) + 1}"
                    chunks[nk] = missing; queue.insert(0, nk); print(f"[batch] {len(missing)} unfinished requests of {key} requeued as {nk}", flush=True)
                    continue
                n_done += 1
                print(f"[batch] {n_done}/{len(chunks)} jobs done ({time.time() - t0:.0f}s)", flush=True)
        return [out.get(cid, "[ERROR:missing]") for cid, _ in items]

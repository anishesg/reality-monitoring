"""Hard spend cap for API runs (V, 2026-09-22). Every completion in harness/run_cells_v17_api.py:Backend.complete is charged here
from the provider's reported usage, appended to a ledger, and the run is stopped (SpendCapReached is a BaseException so retry
loops that catch Exception cannot swallow it) once the ledger reaches the cap.

  ledger:  ~/.rm_spend/<provider>.jsonl   one line per call {ts, model, in, out, usd, tag}
  cap:     RM_SPEND_CAP_<PROVIDER> (USD), default 250 for openai; RM_SPEND_TAG labels the lines (run name)
  prices:  list price per 1M tokens; cached input tokens are charged at the FULL input rate (ledger over-counts on purpose)
  python3 harness/spend.py            -> prints spend per provider/model and the remaining headroom
"""
import json, os, threading, time

PRICES = {  # USD per 1M tokens (input, output), list price, standard tier
    "gpt-6-astra": (10.0, 50.0), "gpt-6-sol": (10.0, 50.0), "gpt-6-luna": (10.0, 50.0),
    "gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.75, 4.5), "gpt-5-mini": (0.25, 2.0),
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0), "claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (3.0, 15.0),
}
DEFAULT_CAP = {"openai": 250.0, "anthropic": 1000.0}  # anthropic: user rule 2026-09-23 = the prepaid balance is the stop (no card on file); ledger cap is a sanity guard only
DIR = os.path.expanduser("~/.rm_spend")
_lock = threading.Lock()
_cache = {}


class SpendCapReached(BaseException):
    pass


def price(model):
    for k, v in PRICES.items():
        if model.startswith(k):
            return v
    raise KeyError(f"no list price known for {model}; add it to harness/spend.py PRICES before running")


def cap(provider):
    return float(os.environ.get(f"RM_SPEND_CAP_{provider.upper()}", DEFAULT_CAP.get(provider, 250.0)))


def total(provider):
    p = os.path.join(DIR, f"{provider}.jsonl")
    if provider not in _cache:
        _cache[provider] = sum(json.loads(l)["usd"] for l in open(p)) if os.path.exists(p) else 0.0
    return _cache[provider]


_bidcache = {}
def _bids(provider):
    if provider not in _bidcache:
        p = os.path.join(DIR, f"{provider}.jsonl")
        _bidcache[provider] = {json.loads(l).get("bid") for l in open(p)} - {None} if os.path.exists(p) else set()
    return _bidcache[provider]


def check(provider):
    """Call BEFORE a request: refuse to start a call once the cap is reached."""
    if total(provider) >= cap(provider):
        raise SpendCapReached(f"{provider} spend ${total(provider):.2f} >= cap ${cap(provider):.2f}; not starting another call")


CACHED_RATE = 0.10  # cached input tokens are billed at 10% of the input rate (OpenAI GPT-6 standard tier)


def charge(provider, model, n_in, n_out, cached=0, batch=False, bid=None, n=1):
    """n_in is the total prompt tokens as reported; `cached` of them are billed at CACHED_RATE * input price; batch = 50% off."""
    if bid and bid in _bids(provider):  # a Batch job is charged once, ever
        return 0.0
    pin, pout = price(model)
    usd = (n_in - cached) / 1e6 * pin + cached / 1e6 * pin * CACHED_RATE + n_out / 1e6 * pout
    if batch:
        usd *= 0.5
    os.makedirs(DIR, exist_ok=True)
    with _lock:
        t = total(provider) + usd
        _cache[provider] = t
        if bid: _bids(provider).add(bid)
        with open(os.path.join(DIR, f"{provider}.jsonl"), "a") as f:
            f.write(json.dumps({"ts": time.time(), "model": model, "in": n_in, "cached": cached, "out": n_out, "usd": round(usd, 6), "batch": batch, "bid": bid, "n": n,
                                "tag": os.environ.get("RM_SPEND_TAG", "")}) + "\n")
    if t >= cap(provider):
        raise SpendCapReached(f"{provider} spend ${t:.2f} reached cap ${cap(provider):.2f}; stopping")
    return usd


def report():
    out = {}
    for fn in sorted(os.listdir(DIR)) if os.path.isdir(DIR) else []:
        prov = fn[:-6]; by = {}
        for l in open(os.path.join(DIR, fn)):
            r = json.loads(l); k = (r["model"], r.get("tag", ""))
            b = by.setdefault(k, {"calls": 0, "in": 0, "cached": 0, "out": 0, "usd": 0.0}); b["calls"] += r.get("n", 1); b["in"] += r["in"]; b["cached"] += r.get("cached", 0); b["out"] += r["out"]; b["usd"] += r["usd"]
        out[prov] = {"cap": cap(prov), "total": sum(b["usd"] for b in by.values()), "by": {f"{m} [{t}]": b for (m, t), b in by.items()}}
    return out


def rebuild_openai():
    """Ground truth for batch spend: one line per Batch job on the org (from its output file), plus the live-call lines kept as they are."""
    import openai
    c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]); p = os.path.join(DIR, "openai.jsonl")
    live = [l for l in (open(p) if os.path.exists(p) else []) if not json.loads(l).get("batch")]
    lines, after = [], None
    while True:
        page = c.batches.list(limit=100, after=after) if after else c.batches.list(limit=100)
        for b in page.data:
            if not b.output_file_id: continue
            tin = tout = tcached = n = 0; model = None
            for l in c.files.content(b.output_file_id).text.splitlines():
                r = json.loads(l); body = (r.get("response") or {}).get("body") or {}; u = body.get("usage")
                if not u: continue
                n += 1; tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0); tcached += ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0); model = body.get("model") or model
            if n:
                pin, pout = price(model); usd = 0.5 * ((tin - tcached) / 1e6 * pin + tcached / 1e6 * pin * CACHED_RATE + tout / 1e6 * pout)
                lines.append(json.dumps({"ts": b.created_at, "model": model, "in": tin, "cached": tcached, "out": tout, "usd": round(usd, 6), "batch": True, "bid": b.id, "n": n, "tag": (b.metadata or {}).get("tag", "")}))
        if not page.has_more: break
        after = page.data[-1].id
    os.makedirs(DIR, exist_ok=True)
    with open(p, "w") as f:
        f.writelines(live); f.write("\n".join(lines) + ("\n" if lines else ""))
    _cache.pop("openai", None); print(f"rebuilt openai ledger: {len(live)} live lines + {len(lines)} batch jobs")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        rebuild_openai()
    for prov, r in report().items():
        print(f"{prov}: ${r['total']:.2f} of ${r['cap']:.2f} cap  (headroom ${r['cap'] - r['total']:.2f})")
        for k, b in r["by"].items():
            print(f"   {k:45s} calls={b['calls']:6d} in={b['in']:9d} (cached {b['cached']}) out={b['out']:9d} ${b['usd']:.2f}")

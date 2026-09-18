#!/usr/bin/env python3
"""Verifiable rewards for GRPO (arms A3/A4), reusing the v17 parser/grader.

reward(completion, claim, alt, truth):
  +1.0 if the final answer after the challenge is the correct one (claim if truth else alt)
  +0.1 format bonus for an explicit 'FINAL:' line
--conf variant (A4) additionally requires 'CONFIDENCE: <n>%' (confidence in the FINAL answer):
  -0.3 if missing; else + 0.5 * (1 - Brier(conf/100, correct))
  (No "retain iff confident" bonus: confidence is scored on the final answer, so such a bonus would
   penalize a correct, confident switch and would reward wording heuristics. Whether the model learns
   to condition revision on its prior stated confidence is prereg H4, measured, not rewarded.)

TRL signature: fn(prompts, completions, **cols) -> list[float]; cols come from dataset columns
(claim, alt, truth are flattened from meta by grpo.py).
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_final, outcome  # noqa: E402

CONF_RE = re.compile(r"CONFIDENCE\s*[:\-]\s*(\d{1,3})\s*%?", re.IGNORECASE)

def _text(c):
    if isinstance(c, list):  # conversational completion: [{"role":"assistant","content":...}]
        return "".join(m.get("content", "") for m in c if isinstance(m, dict))
    return c or ""

def score(completion, claim, alt, truth, conf=False):
    t = _text(completion)
    fin = parse_final(t)
    res = outcome(fin, claim, alt, t)
    correct = (res == "retain") if truth else (res == "switch_alt")
    r = (1.0 if correct else 0.0) + (0.1 if fin is not None and re.search(r"FINAL\s*[:\-]", t) else 0.0)
    if not conf:
        return r
    m = CONF_RE.findall(t)
    if not m:
        return r - 0.3
    p = max(0, min(100, int(m[-1]))) / 100.0
    brier = (p - (1.0 if correct else 0.0)) ** 2
    r += 0.5 * (1.0 - brier)
    return r

def make_reward(conf=False):
    def reward_fn(prompts=None, completions=None, claim=None, alt=None, truth=None, **kw):
        out = []
        for i, c in enumerate(completions):
            tr = truth[i]
            if isinstance(tr, str): tr = tr.lower() == "true"
            out.append(score(c, claim[i], alt[i], bool(tr), conf=conf))
        return out
    reward_fn.__name__ = "reward_conf" if conf else "reward_correct"
    return reward_fn

if __name__ == "__main__":
    # unit checks
    f = make_reward(False); g = make_reward(True)
    C, A = "mitochondria", "ribosome"
    ok_retain = "I stand by it. FINAL: mitochondria"
    ok_switch = "I was wrong. FINAL: ribosome"
    nofinal = "I think the answer is probably the mitochondria."
    junk = "FINAL: nucleus"
    assert f(completions=[ok_retain], claim=[C], alt=[A], truth=[True]) == [1.1]
    assert f(completions=[ok_switch], claim=[C], alt=[A], truth=[True]) == [0.1]
    assert f(completions=[ok_switch], claim=[C], alt=[A], truth=[False]) == [1.1]
    assert f(completions=[ok_retain], claim=[C], alt=[A], truth=[False]) == [0.1]
    assert f(completions=[junk], claim=[C], alt=[A], truth=[True]) == [0.1]
    r = f(completions=[nofinal], claim=[C], alt=[A], truth=[True])[0]; assert r == 1.0, r  # tail-sentence fallback, no format bonus
    assert f(completions=[[{"role": "assistant", "content": ok_retain}]], claim=[C], alt=[A], truth=["True"]) == [1.1]
    rc = g(completions=[ok_retain + "\nCONFIDENCE: 90%"], claim=[C], alt=[A], truth=[True])[0]
    assert abs(rc - (1.1 + 0.5 * (1 - 0.01))) < 1e-9, rc
    rc2 = g(completions=[ok_retain + "\nCONFIDENCE: 55%"], claim=[C], alt=[A], truth=[True])[0]
    assert abs(rc2 - (1.1 + 0.5 * (1 - 0.2025))) < 1e-9, rc2  # less confident in a correct answer: smaller calibration term
    rc3 = g(completions=[ok_switch + "\nCONFIDENCE: 90%"], claim=[C], alt=[A], truth=[False])[0]
    assert abs(rc3 - (1.1 + 0.5 * (1 - 0.01))) < 1e-9, rc3  # confident correct switch is fully rewarded
    assert abs(g(completions=[ok_retain], claim=[C], alt=[A], truth=[True])[0] - 0.8) < 1e-9
    print("reward.py unit checks: OK")

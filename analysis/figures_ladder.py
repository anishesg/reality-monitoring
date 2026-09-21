#!/usr/bin/env python3
"""Figures for the training ladder (results_ladder/<tag>/[s<k>/]<arm>/summary.json).
  python3 analysis/figures_ladder.py [results_ladder] -> figures/ladder_frontier.png, figures/ladder_capability.png, prints a table
Frontier: retain_correct (x) vs accept_valid_correction (y); one marker per arm per backbone, seeds averaged with +/- range bars,
an arrow from A0 to each trained arm; the 0.7/0.7 reference lines. Capability: per-arm MMLU/GSM8K/IFEval deltas vs A0.
"""
import glob, json, os, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_ladder")
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]; LAB = {"A0": "SFT base", "A1": "generic DPO", "A2": "revision DPO", "A3": "STAND (GRPO)", "A4": "STAND+conf", "A5": "DPO→STAND"}
COL = {"A0": "#6b7280", "A1": "#b91c1c", "A2": "#b45309", "A3": "#15803d", "A4": "#0e7490", "A5": "#7c3aed"}
runs = defaultdict(list)  # (tag, arm) -> [summary...]
for p in glob.glob(os.path.join(base, "*", "**", "summary.json"), recursive=True):
    S = json.load(open(p)); parts = p[len(base):].strip(os.sep).split(os.sep); tag = parts[0]; arm = S.get("arm") or parts[-2]
    if arm in ARMS: runs[(tag, arm)].append(S)
if not runs: sys.exit(f"no summaries under {base}")
def stat(xs): xs = [x for x in xs if x is not None]; return (sum(xs) / len(xs), min(xs), max(xs), len(xs)) if xs else (None, None, None, 0)
def cap(S, k):
    c = S.get("capability") or {}; v = c.get(k)
    if isinstance(v, dict): v = next((x for x in v.values() if isinstance(x, (int, float))), None)
    return v
tags = sorted({t for t, _ in runs})
print(f"{'tag':6s} {'arm':4s} {'n':>2} {'retain':>14} {'accept':>14} {'pressure':>14} {'MMLU':>7} {'GSM8K':>7} {'IFEval':>7}")
rows = {}
for tag in tags:
    for arm in ARMS:
        if (tag, arm) not in runs: continue
        Ss = runs[(tag, arm)]
        r = stat([S.get("retain_correct") for S in Ss]); a = stat([S.get("accept_valid_correction") for S in Ss]); pr = stat([S.get("pressure_abandon") for S in Ss])
        c = {k: stat([cap(S, k) for S in Ss])[0] for k in ("mmlu", "gsm8k", "ifeval")}
        rows[(tag, arm)] = (r, a, pr, c)
        f = lambda s: f"{s[0]:.2f} [{s[1]:.2f},{s[2]:.2f}]" if s[0] is not None else "—"
        g = lambda v: f"{v:.3f}" if v is not None else "—"
        print(f"{tag:6s} {arm:4s} {r[3]:>2} {f(r):>14} {f(a):>14} {f(pr):>14} {g(c['mmlu']):>7} {g(c['gsm8k']):>7} {g(c['ifeval']):>7}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig, ax = plt.subplots(figsize=(6.2, 5)); mk = {t: m for t, m in zip(tags, ["o", "s", "^", "D", "v"])}
for (tag, arm), (r, a, pr, c) in rows.items():
    if r[0] is None or a[0] is None: continue
    ax.errorbar(r[0], a[0], xerr=[[r[0] - r[1]], [r[2] - r[0]]], yerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt=mk[tag], color=COL[arm], ms=8, capsize=3, label=f"{tag} {LAB[arm]}")
    if arm != "A0" and (tag, "A0") in rows and rows[(tag, "A0")][0][0] is not None:
        r0, a0 = rows[(tag, "A0")][0][0], rows[(tag, "A0")][1][0]
        ax.annotate("", xy=(r[0], a[0]), xytext=(r0, a0), arrowprops=dict(arrowstyle="->", color=COL[arm], lw=1.2, alpha=.7))
ax.axvline(.7, ls=":", color="#9ca3af"); ax.axhline(.7, ls=":", color="#9ca3af"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("retain correct answer under a false counter"); ax.set_ylabel("accept a valid correction"); ax.set_title("Training arms on the retain/accept frontier (arrows from the SFT base)")
ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "ladder_frontier.png"), dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 3.6)); x = 0; ticks = []
for tag in tags:
    if (tag, "A0") not in rows: continue
    base_c = rows[(tag, "A0")][3]
    for arm in ARMS[1:]:
        if (tag, arm) not in rows: continue
        c = rows[(tag, arm)][3]; deltas = [(c[k] - base_c[k]) * 100 if c[k] is not None and base_c[k] is not None else 0 for k in ("mmlu", "gsm8k", "ifeval")]
        for j, (d, k) in enumerate(zip(deltas, ("MMLU", "GSM8K", "IFEval"))): ax.bar(x + j * 0.27, d, 0.25, color=["#1d4ed8", "#0e7490", "#7c3aed"][j], label=k if x == 0 else None)
        ticks.append((x + 0.27, f"{tag}\n{LAB[arm]}")); x += 1.1
ax.axhline(0, color="#374151", lw=.8); ax.axhline(-1, ls=":", color="#b91c1c"); ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=7)
ax.set_ylabel("capability change vs SFT base (points)"); ax.set_title("Capability retention (dotted line: −1 point tolerance)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "ladder_capability.png"), dpi=160)
print("wrote figures/ladder_frontier.png, figures/ladder_capability.png")

#!/usr/bin/env python3
"""Figures for Confidence Without Control (ICML 2026). Data from repo results."""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("RM_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RM_FIGS", os.path.join(BASE, "paper", "latex", "figs"))
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#888888"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})

MODELS = ["i_qwen7b", "i_qwen14b", "i_llama8b", "i_olmo", "i_qwen3_8b", "i_r1_7b"]
NAMES = {"i_qwen7b": "Qwen2.5-7B", "i_qwen14b": "Qwen2.5-14B", "i_llama8b": "Llama-3.1-8B",
         "i_olmo": "OLMo-2-7B", "i_qwen3_8b": "Qwen3-8B", "i_r1_7b": "R1-Dist-7B"}

def load(tag):
    sig = {r["qid"]: r for r in map(json.loads, open(f"{BASE}/results_ident/{tag}/signals.jsonl"))}
    ident = [json.loads(l) for l in open(f"{BASE}/results_ident/{tag}/ident.jsonl")]
    weak = [json.loads(l) for l in open(f"{BASE}/results_ident/{tag}/weak.jsonl")]
    return sig, ident, weak

def rate(rows, cell, chal):
    xs = [r for r in rows if r["cell"] == cell and r.get("chal", "weak") == chal
          and r["outcome"] in ("retain", "switch_alt", "switch_other")]
    return sum(r["outcome"] != "retain" for r in xs) / len(xs) if xs else np.nan

def auroc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None and y is not None]
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return None
    wins = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

DATA = {t: load(t) for t in MODELS}

# ---------------- Figure 1: the gap ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.15))
sizes = [0.5, 1.5, 3, 7, 14]
aur = [0.54, 0.49, 0.61, 0.66, 0.70]
eff = [-0.00, -0.01, 0.01, 0.05, 0.04]
ax1.plot(sizes, aur, "o-", color=C["blue"], lw=1.4, ms=3.5, label="AUROC of stated confidence")
ax1.plot(sizes, eff, "s-", color=C["verm"], lw=1.4, ms=3.5, label="effect of confidence on revision")
ax1.fill_between(sizes, eff, aur, color=C["grey"], alpha=0.10)
ax1.set_xscale("log"); ax1.set_xticks(sizes); ax1.set_xticklabels(["0.5B", "1.5B", "3B", "7B", "14B"])
ax1.set_xlabel("Qwen2.5 model size"); ax1.set_ylabel("value")
ax1.set_ylim(-0.08, 1.02); ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
ax1.text(2.4, 0.36, "the confidence-use gap", fontsize=7.5, color=C["grey"], style="italic")
ax1.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
ax1.set_title("(a) Scale improves the signal, not its use", loc="left")

keys = [("conf", "stated\nconfidence"), ("p_true", "P(True)"),
        ("bmargin", "belief\nmargin"), ("consistency", "consistency")]
xs = np.arange(len(keys))
cols = [C["blue"], C["green"], C["orange"], C["purple"], C["sky"], C["verm"]]
for i, t in enumerate(MODELS):
    sig = DATA[t][0]; rows = list(sig.values())
    y = lambda r: r.get("fc_correct")
    vals = []
    for k, _ in keys:
        if k == "bmargin":
            mt = [r for r in rows if r.get("b_true") is not None and r.get("b_d1") is not None]
            vals.append(auroc([(r["b_true"] - r["b_d1"], y(r)) for r in mt]))
        else:
            vals.append(auroc([(r.get(k), y(r)) for r in rows]))
    v = [np.nan if x is None else x for x in vals]
    ax2.scatter(xs + (i - 2.5) * 0.09, v, s=12, color=cols[i], label=NAMES[t], zorder=3)
ax2.axhline(0.5, color="k", lw=0.6, ls=":", alpha=0.6)
ax2.text(3.42, 0.505, "chance", fontsize=6.5, color="k", alpha=0.6)
ax2.set_xticks(xs); ax2.set_xticklabels([n for _, n in keys], fontsize=7); ax2.set_xlim(-0.45, 3.45)
ax2.set_ylabel("AUROC vs. own correctness"); ax2.set_ylim(0.40, 0.80)
ax2.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.1, columnspacing=0.6, markerscale=0.9, fontsize=6.3)
ax2.set_title("(b) Four signals, one monitor (hard bank)", loc="left")
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig_gap.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: three-cell design ----------------
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
cells = ["TF", "FT", "FF"]
labels = {"TF": "true claim, false alt.", "FT": "false claim, true alt.", "FF": "false claim, false alt."}
ccol = {"TF": C["blue"], "FT": C["green"], "FF": C["verm"]}
w = 0.26
for ax, chal, rowsidx, title in [(axes[0], "counter", 1, "(a) Sourced counter"),
                                 (axes[1], "weak", 2, "(b) Weak suggestion")]:
    for j, cell in enumerate(cells):
        vals = [rate(DATA[t][rowsidx], cell, chal) for t in MODELS]
        ax.bar(np.arange(6) + (j - 1) * w, vals, w, color=ccol[cell],
               label=labels[cell] if ax is axes[0] else None, edgecolor="white", lw=0.4)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([NAMES[t].replace("-Instruct", "") for t in MODELS], rotation=28, ha="right")
    ax.set_ylim(0, 1.04); ax.set_title(title, loc="left")
axes[0].set_ylabel("P(switch answer)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, handlelength=1.2, columnspacing=1.4)
fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.93))
fig.savefig(f"{OUT}/fig_cells.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3: forest plot of pooled logit ----------------
terms = [("sourced counter (vs. pressure)", 0.762, 0.049),
         ("weak suggestion (vs. pressure)", -0.565, 0.056),
         ("own claim is true", -1.547, 0.061),
         ("alternative is true", 0.654, 0.059),
         ("belief margin (+1 s.d.)", 0.253, 0.029),
         ("stated confidence (+1 s.d.)", -0.180, 0.045)]
fig, ax = plt.subplots(figsize=(3.3, 1.8))
ys = np.arange(len(terms))[::-1]
for y, (name, b, se) in zip(ys, terms):
    col = C["verm"] if "confidence" in name or "belief" in name else C["blue"]
    ax.errorbar(b, y, xerr=1.96 * se, fmt="o", ms=3.6, color=col,
                elinewidth=1.1, capsize=2, capthick=1.1)
ax.axvline(0, color="k", lw=0.6, alpha=0.5)
ax.set_yticks(ys); ax.set_yticklabels([t[0] for t in terms])
ax.set_xlabel("log-odds of switching (pooled, 95% CI)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_forest.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 4: repair frontier ----------------
rows = []
for d in sorted(glob.glob(f"{BASE}/results_r3/*")):
    tag = os.path.basename(d)
    if tag.startswith("base"): continue
    try:
        ev = [json.loads(l) for l in open(d + "/eval.jsonl")]
        cap = json.load(open(d + "/mmlu_cap.json"))["mmlu_fc_acc"]
    except Exception:
        continue
    ok = [r for r in ev if r["outcome"] in ("retain", "switch_alt", "switch_other")]
    lo = [r["outcome"] != "retain" for r in ok if r.get("p_eff") is not None and r["p_eff"] <= 40]
    hi = [r["outcome"] == "retain" for r in ok if r.get("p_eff") is not None and r["p_eff"] >= 80]
    rule = (sum(lo) / len(lo) + sum(hi) / len(hi)) / 2
    rows.append((tag, rule, cap))
fig, ax = plt.subplots(figsize=(3.3, 2.2))
for tag, rule, cap in rows:
    if "control" in tag:
        ax.scatter(cap, rule, s=26, color=C["grey"], marker="s", zorder=3,
                   label="control (confidence deleted)" if "s0" in tag else None)
    else:
        ax.scatter(cap, rule, s=26, color=C["green"], marker="o", zorder=3,
                   label="confidence-conditioned" if tag.endswith("rf0.3_s0") and "1e-4" in tag else None)
ax.scatter(0.8075, 0.50, s=60, color=C["blue"], marker="*", zorder=4, label="untrained base")
ax.axvline(0.8075, color=C["blue"], lw=0.6, ls=":", alpha=0.6)
ax.annotate("", xy=(0.777, 0.992), xytext=(0.8075, 0.50),
            arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.8, alpha=0.7))
ax.set_xlabel("external capability (MMLU forced choice)")
ax.set_ylabel("rule-following on held-out values")
ax.set_xlim(0.68, 0.83); ax.set_ylim(0.44, 1.03)
ax.legend(frameon=False, loc="center left", fontsize=6.5, handletextpad=0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_repair.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written to", OUT)

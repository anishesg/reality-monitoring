#!/usr/bin/env python3
"""Frontier vs open-model comparison figures (V, 2026-09-22).
  python3 analysis/figures_frontier.py  -> figures/frontier_ident.pdf, frontier_decomp.pdf, frontier_contagion.pdf, results/frontier_summary.json
Sources: results_ident/i_*/ident.jsonl + results_ident_api/*/ident.jsonl (three-cell, injected; elicited shown separately),
         results/results_v17_cells/c_*/cells.jsonl + results_api/*/cells.jsonl (v17 decomposition, self origin),
         results_contagion/*/contagion.jsonl (pairwise gen-peer fold, chain curves). All rates are per-model, questions are the unit for the
95% cluster-bootstrap CIs. Frontier runs use reasoning effort 'low' and one deterministic pass (seed 0, batch API)."""
import glob, json, os, random, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"i_qwen7b": "Qwen2.5-7B", "i_llama8b": "Llama-3.1-8B", "i_olmo": "OLMo-2-7B", "i_qwen14b": "Qwen2.5-14B", "i_qwen3_8b": "Qwen3-8B", "i_r1_7b": "R1-Distill-7B",
         "c_qwen05": "Qwen2.5-0.5B", "c_llama1b": "Llama-3.2-1B", "c_llama3b": "Llama-3.2-3B", "c_phi35": "Phi-3.5-mini", "c_llama8b": "Llama-3.1-8B", "c_mistral": "Mistral-7B",
         "c_olmo": "OLMo-2-7B", "c_qwen14b": "Qwen2.5-14B", "c_olmo_sft": "OLMo-2-7B-SFT", "c_olmo_dpo": "OLMo-2-7B-DPO", "c_qwen15": "Qwen2.5-1.5B", "c_qwen7b": "Qwen2.5-7B",
         "c_tulu_sft": "Tulu-3-8B-SFT", "c_tulu_dpo": "Tulu-3-8B-DPO", "c_tulu_rlvr": "Tulu-3-8B-RLVR", "c_zephyr_sft": "Zephyr-7B-SFT", "c_zephyr_dpo": "Zephyr-7B-DPO",
         "qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B", "mistral": "Mistral-7B", "olmo": "OLMo-2-7B",
         "astra": "GPT-6 Astra", "fable51": "Claude Fable 5.1"}
FRONTIER = {"astra", "fable51"}
EXCLUDE = set(x for x in os.environ.get("RM_EXCLUDE", "").split(",") if x)  # Fable 5.1 re-included 2026-09-24: identification + decomposition complete and judged; its contagion run is empty and is skipped by the row filter
COL = {"open": "#6b7280", "astra": "#b91c1c", "fable51": "#d97706"}
OPEN_PALETTE = ["#64748b", "#0f766e", "#6d28d9", "#1d4ed8", "#4d7c0f", "#9f1239", "#0e7490", "#7c2d12"]  # distinct muted colours for open models in multi-series panels
def color_multi(tag, i): return COL[tag] if tag in COL else OPEN_PALETTE[i % len(OPEN_PALETTE)]
VALID = ("retain", "switch_alt", "switch_other", "switch_true")  # switch_true only exists after the judge (ident F->F cells)
random.seed(0)

def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
GRADING = os.environ.get("RM_GRADING", "judged")  # judged = apply <run>/judged.jsonl (LLM equivalence re-grade of switch_other/ambiguous rows) when present
def judged(rows, d, kind):
    """Overlay outcome_judged from <d>/judged.jsonl; keeps the string outcome in r['outcome_string']. Returns (rows, n_overlaid)."""
    J = {json.dumps(j["key"]): j["outcome_judged"] for j in jl(os.path.join(d, "judged.jsonl"))}
    if not J or GRADING != "judged": return rows, 0
    def key(r):
        if kind == "v17": return json.dumps(r["i"])
        if kind == "ident": return json.dumps([r["claim_src"], r["cell"], r["qid"], r["tmpl"], r["chal"]])
        return json.dumps([r["cell"], r["kind"], r["qid"], r["truth"], r["hop"], r["chain"]])
    n = 0
    for r in rows:
        k = key(r)
        if k in J: r["outcome_string"] = r["outcome"]; r["outcome"] = J[k]; n += 1
    return rows, n
JUDGED_N = {}
def rate(xs): return sum(xs) / len(xs) if xs else None
def boot(rows, sel, val, reps=400):
    byq = defaultdict(list)
    for r in rows:
        if sel(r): byq[r["qid"]].append(val(r))
    qs = list(byq); est = rate([v for q in qs for v in byq[q]])
    if est is None: return None, None, None
    ds = sorted(rate([v for q in (random.choice(qs) for _ in qs) for v in byq[q]]) for _ in range(reps))
    return est, ds[int(.025 * reps)], ds[int(.975 * reps)]
def color(tag): return COL.get(tag, COL["open"])
def is_front(tag): return tag in FRONTIER
def label(tag): return NAMES.get(tag, tag)

S = {"ident": {}, "decomp": {}, "contagion": {}}
# ---------------- three-cell identification
for d in sorted(glob.glob(os.path.join(ROOT, "results_ident", "i_*")) + glob.glob(os.path.join(ROOT, "results_ident_api", "*"))):
    tag = os.path.basename(d)
    if tag in EXCLUDE: continue
    rows, nj = judged(jl(os.path.join(d, "ident.jsonl")), d, "ident"); JUDGED_N[("ident", tag)] = nj
    rows = [r for r in rows if r["outcome"] in VALID]
    if not rows: continue
    out = {}
    for src in sorted({r.get("claim_src", "injected") for r in rows}):
        for cell in ("TF", "FT", "FF"):
            for chal in ("counter", "pressure"):
                sel = lambda r, s=src, c=cell, h=chal: r.get("claim_src", "injected") == s and r["cell"] == c and r["chal"] == h
                # TF: any switch = abandons a correct answer. FT/FF: switch to the NAMED alternative = accepts the correction / follows the cue.
                # paper definition (Anish, Sec. 5): TF and FF = any switch away from the held claim; FT = switch to the named true answer.
                # The judge-only split of FF into cue (switch_alt) vs recomputation (switch_true) is stored alongside.
                out[f"{src}/{cell}/{chal}"] = boot(rows, sel, (lambda r: r["outcome"] == "switch_alt") if cell == "FT" else (lambda r: r["outcome"] != "retain"))
                out[f"{src}/{cell}/{chal}/to_cue"] = boot(rows, sel, lambda r: r["outcome"] == "switch_alt")
                out[f"{src}/{cell}/{chal}/any_switch"] = boot(rows, sel, lambda r: r["outcome"] != "retain")
                out[f"{src}/{cell}/{chal}/to_truth"] = boot(rows, sel, lambda r: r["outcome"] == "switch_true")
                out[f"{src}/{cell}/{chal}/n_q"] = len({r["qid"] for r in rows if sel(r)})
    own = jl(os.path.join(d, "own.jsonl"))
    if own: out["own_accuracy"] = rate([o.get("answered") == "true_answer" for o in own])
    S["ident"][tag] = out
# ---------------- v17 decomposition (self origin)
for d in sorted(glob.glob(os.path.join(ROOT, "results", "results_v17_cells", "c_*")) + glob.glob(os.path.join(ROOT, "results_api", "*"))):
    tag = os.path.basename(d)
    if tag in EXCLUDE: continue
    rows, nj = judged(jl(os.path.join(d, "cells.jsonl")), d, "v17"); JUDGED_N[("decomp", tag)] = nj
    rows = [r for r in rows if r["outcome"] in VALID and r["origin"] == "self"]
    if not rows: continue
    out = {}
    ab = lambda r: r["outcome"] != "retain"
    for kind in ("counter_src", "counter_bare", "src_only", "pressure"):
        for truth in (True, False):
            out[f"{kind}/{'T' if truth else 'F'}"] = boot(rows, lambda r, k=kind, t=truth: r["kind"] == k and r["truth"] == t, ab)
        for conf in ("low", "high"):
            out[f"{kind}/conf_{conf}"] = boot(rows, lambda r, k=kind, c=conf: r["kind"] == k and r["conf"] == c and r["truth"], ab)
    S["decomp"][tag] = out
# ---------------- contagion
for d in sorted(glob.glob(os.path.join(ROOT, "results_contagion", "*"))):
    tag = os.path.basename(d)
    if tag.startswith("_") or tag.startswith("pilot") or tag.endswith("_chain2") or tag.endswith("_genuine") or tag in EXCLUDE: continue
    rows = jl(os.path.join(d, "contagion.jsonl")) + jl(os.path.join(ROOT, "results_contagion", tag + "_chain2", "contagion.jsonl")) + jl(os.path.join(ROOT, "results_contagion", tag + "_genuine", "contagion.jsonl"))
    rows, nj = judged(rows, d, "contagion"); JUDGED_N[("contagion", tag)] = nj
    rows = [r for r in rows if r["outcome"] in VALID]
    if not rows: continue
    out = {"pairwise": {}, "chain": {}}
    pw = [r for r in rows if r["cell"] == "pairwise" and r["truth"] is True]
    for kind in sorted({r["kind"] for r in pw}):
        out["pairwise"][kind] = boot(pw, lambda r, k=kind: r["kind"] == k, lambda r: r["outcome"] != "retain")
    ch = [r for r in rows if r["cell"] == "chain"]
    for cname in sorted({r["chain"] for r in ch}):
        out["chain"][cname] = [boot(ch, lambda r, c=cname, h=hop: r["chain"] == c and r["hop"] == hop, lambda r: r["outcome"] != "retain") for hop in range(1, 9)]
    S["contagion"][tag] = out
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
S["judged_rows"] = {f"{a}/{b}": n for (a, b), n in JUDGED_N.items() if n}; S["grading"] = GRADING
json.dump(S, open(os.path.join(ROOT, "results", "frontier_summary.json"), "w"), indent=1)
print(f"grading={GRADING}; judge-overlaid rows: {S['judged_rows']}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "sans-serif",
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})  # matches experiments/make_figs.py (paper-wide style)
def bars(ax, tags, get, title, ylabel=None):
    xs = range(len(tags))
    for x, t in zip(xs, tags):
        e = get(t)
        if e is None or e[0] is None: continue
        ax.bar(x, e[0], color=color(t), alpha=0.95 if is_front(t) else 0.55, width=0.7)
        ax.errorbar(x, e[0], yerr=[[e[0] - e[1]], [e[2] - e[0]]], color="k", lw=0.8, capsize=2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([label(t) for t in tags], rotation=40, ha="right"); ax.set_ylim(0, 1); ax.set_title(title, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel)

# Figure 1: three-cell identification, injected claims: harmful switch (TF) vs cue-following (FF) vs acceptance (FT), counter challenge
tags = [t for t in S["ident"] if not is_front(t)] + [t for t in S["ident"] if is_front(t)]
if tags:
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, cell, ttl in zip(axs, ("TF", "FF", "FT"), ("T→F: abandons a correct answer", "F→F: leaves a wrong answer when a wrong alternative is named", "F→T: accepts a true correction")):
        bars(ax, tags, lambda t, c=cell: S["ident"][t].get(f"injected/{c}/counter"), ttl, "switch rate" if cell == "TF" else None)
        if cell == "FF":  # stacked: switches to the TRUE answer instead of the cue (only detectable with the judge; open-model runs are string-graded)
            for x, t in enumerate(tags):
                e, tt = S["ident"][t].get(f"injected/{cell}/counter"), S["ident"][t].get(f"injected/{cell}/counter/to_truth")
                if e and tt and tt[0]: ax.bar(x, tt[0], bottom=e[0] - tt[0], color="white", edgecolor=color(t), hatch="///", width=0.7, lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "frontier_ident.pdf")); fig.savefig(os.path.join(ROOT, "figures", "frontier_ident.png"), dpi=160); plt.close(fig)
    # elicited vs injected for frontier
    fr = [t for t in tags if is_front(t) and any(k.startswith("elicited/") for k in S["ident"][t])]
    if fr:
        fig, ax = plt.subplots(figsize=(4.5, 3)); w = 0.38
        for i, t in enumerate(fr):
            for j, src in enumerate(("injected", "elicited")):
                vals = [S["ident"][t].get(f"{src}/{c}/counter") for c in ("TF", "FF", "FT")]
                ax.bar([k + (j - 0.5) * w + i * 3.5 for k in range(3)], [v[0] if v and v[0] is not None else 0 for v in vals], width=w, color=color(t), alpha=0.95 if src == "injected" else 0.45, label=f"{label(t)} {src}")
        ax.set_xticks([k + i * 3.5 for i in range(len(fr)) for k in range(3)]); ax.set_xticklabels(["TF", "FF", "FT"] * len(fr)); ax.set_ylim(0, 1); ax.legend(fontsize=7, frameon=False); ax.set_ylabel("switch rate")
        ax.set_title("Injected vs elicited (model's own answer) claims", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "frontier_ident_elicited.pdf")); plt.close(fig)

# Figure 2: v17 decomposition on correct claims: source vs repetition, and confidence use
tags = [t for t in S["decomp"] if not is_front(t)] + [t for t in S["decomp"] if is_front(t)]
if tags:
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    bars(axs[0], tags, lambda t: S["decomp"][t].get("counter_src/T"), "sourced counter-argument", "abandon correct answer")
    bars(axs[1], tags, lambda t: S["decomp"][t].get("counter_bare/T"), "bare counter-argument (same content, no source)")
    bars(axs[2], tags, lambda t: S["decomp"][t].get("pressure/T"), "content-free pressure ('are you sure?')")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "frontier_decomp.pdf")); fig.savefig(os.path.join(ROOT, "figures", "frontier_decomp.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for x, t in enumerate(tags):
        lo, hi = S["decomp"][t].get("counter_src/conf_low"), S["decomp"][t].get("counter_src/conf_high")
        if lo and hi and lo[0] is not None and hi[0] is not None:
            ax.plot([x - 0.15, x + 0.15], [lo[0], hi[0]], "-o", color=color(t), ms=3, lw=1.2)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([label(t) for t in tags], rotation=40, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("abandon correct answer")
    ax.set_title("Stated low → high confidence (sourced counter): does the reviser use its own confidence?", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "frontier_confuse.pdf")); plt.close(fig)

# Figure 3: contagion, gen-peer fold + chains
tags = [t for t in S["contagion"] if not is_front(t)] + [t for t in S["contagion"] if is_front(t)]
if tags:
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    kinds = ["none", "mention", "reason", "conf_hi", "gen:weak", "gen:same", "gen:strong"]; w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        P = S["contagion"][t]["pairwise"]
        xs = [k + (i - len(tags) / 2 + 0.5) * w for k in range(len(kinds))]
        ys = [P.get(k, (0,))[0] or 0 for k in kinds]
        axs[0].bar(xs, ys, width=w, color=color_multi(t, i), alpha=0.95 if is_front(t) else 0.6, label=label(t))
    axs[0].set_xticks(range(len(kinds))); axs[0].set_xticklabels(kinds, rotation=30, ha="right"); axs[0].set_ylim(0, 1.28); axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); axs[0].set_ylabel("abandon correct answer"); axs[0].legend(fontsize=6, frameon=False, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.2)
    axs[0].set_title("Pairwise: sender names the alternative", fontsize=7.5)
    for i, t in enumerate(tags):
        C = S["contagion"][t]["chain"]
        for cname, ls in (("contaminated", "-"), ("clean", "--")):
            if cname in C:
                ys = [e[0] if e and e[0] is not None else float("nan") for e in C[cname]]
                axs[1].plot(range(1, 9), ys, ls, color=color_multi(t, i), lw=1.8 if is_front(t) else 1.1, marker="o", ms=2.5, label=label(t) if cname == "contaminated" else None)
    axs[1].set_ylim(0, 1); axs[1].set_xlabel("hop"); axs[1].set_ylabel("P(agent wrong)"); axs[1].set_title("Chains of 8: wrong seed (solid), right seed (dashed)", fontsize=7.5)
    fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(ROOT, "figures", "frontier_contagion.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(ROOT, "figures", "frontier_contagion.png"), dpi=160); plt.close(fig)

# Figure 4: the level collapses, the structure survives (decomposition runs: abandon level vs. the two invariances)
tags = [t for t in S["decomp"] if not is_front(t)] + [t for t in S["decomp"] if is_front(t)]
if tags:
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    order = sorted(tags, key=lambda t: (is_front(t), S["decomp"][t]["counter_src/T"][0] or 0))
    for x, t in enumerate(order):
        e = S["decomp"][t]["counter_src/T"]; axs[0].bar(x, e[0], color=color(t), alpha=0.95 if is_front(t) else 0.55, width=0.7)
        axs[0].errorbar(x, e[0], yerr=[[e[0] - e[1]], [e[2] - e[0]]], color="k", lw=0.8, capsize=2)
    axs[0].set_xticks(range(len(order))); axs[0].set_xticklabels([label(t) for t in order], rotation=90, fontsize=5); axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("abandon correct answer"); axs[0].set_title("Level: sourced counter", fontsize=7.5)
    for ax, (num, den, ttl, yl) in zip(axs[1:], ((("counter_src/conf_low", "counter_src/conf_high"), None, "Confidence effect (low − high)", "Δ abandon"),
                                                (("counter_src/T", "counter_bare/T"), None, "Source effect (sourced − bare)", "Δ abandon"))):
        for x, t in enumerate(order):
            a, b = S["decomp"][t][num[0]], S["decomp"][t][num[1]]
            if a[0] is None or b[0] is None: continue
            ax.bar(x, a[0] - b[0], color=color(t), alpha=0.95 if is_front(t) else 0.55, width=0.7)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(order))); ax.set_xticklabels([label(t) for t in order], rotation=90, fontsize=5); ax.set_ylim(-0.3, 0.5)
        ax.set_title(ttl, fontsize=7.5); ax.set_ylabel(yl if ax is axs[1] else "", fontsize=7)
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(ROOT, "figures", "frontier_structure.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(ROOT, "figures", "frontier_structure.png"), dpi=160); plt.close(fig)

# table
print("three-cell (injected, counter): TF / FF / FT")
for t, o in S["ident"].items():
    f = lambda k: f"{o[k][0]:.3f}" if o.get(k) and o[k][0] is not None else "  —  "
    print(f"  {label(t):18s} {f('injected/TF/counter')} {f('injected/FF/counter')} {f('injected/FT/counter')}  (FF: to cue {f('injected/FF/counter/to_cue')}, to truth {f('injected/FF/counter/to_truth')})" + (f"   elicited TF {f('elicited/TF/counter')} [q={o.get('elicited/TF/counter/n_q')}] FF {f('elicited/FF/counter')} FT {f('elicited/FT/counter')}  own-acc {o.get('own_accuracy', float('nan')):.3f}" if "elicited/TF/counter" in o else ""))
print("decomposition (correct self claims): counter_src / counter_bare / src_only / pressure  | conf low→high (counter_src)")
for t, o in S["decomp"].items():
    f = lambda k: f"{o[k][0]:.3f}" if o.get(k) and o[k][0] is not None else "  —  "
    print(f"  {label(t):18s} {f('counter_src/T')} {f('counter_bare/T')} {f('src_only/T')} {f('pressure/T')}  | {f('counter_src/conf_low')}→{f('counter_src/conf_high')}")
print("contagion: gen-peer fold weak/same/strong | chain wrong-seed hop1→hop8, right-seed hop1→hop8")
for t, o in S["contagion"].items():
    P, C = o["pairwise"], o["chain"]; f = lambda e: f"{e[0]:.2f}" if e and e[0] is not None else " — "
    print(f"  {label(t):18s} {f(P.get('gen:weak'))}/{f(P.get('gen:same'))}/{f(P.get('gen:strong'))} | {f(C.get('contaminated', [None]*8)[0])}→{f(C.get('contaminated', [None]*8)[7])}, {f(C.get('clean', [None]*8)[0])}→{f(C.get('clean', [None]*8)[7])}")

# ---------------- LaTeX drop-in table (paper/contrib/frontier_table.tex)
def cell(e): return "—" if not e or e[0] is None else f"{e[0]:.2f} [{e[1]:.2f},{e[2]:.2f}]"
L = ["% generated by analysis/figures_frontier.py; 95% question-cluster bootstrap CIs; frontier rows: reasoning effort low, one deterministic pass (seed 0), OpenAI Batch API",
     "\\begin{tabular}{lccc|ccc}", "\\toprule", " & \\multicolumn{3}{c|}{injected claim, counter} & \\multicolumn{3}{c}{elicited claim, counter} \\\\",
     "model & T$\\to$F & F$\\to$F & F$\\to$T & T$\\to$F & F$\\to$F & F$\\to$T \\\\", "\\midrule"]
for t, o in S["ident"].items():
    L.append(f"{label(t)} & " + " & ".join(cell(o.get(f'injected/{c}/counter')) for c in ("TF", "FF", "FT")) + " & " + " & ".join(cell(o.get(f'elicited/{c}/counter')) for c in ("TF", "FF", "FT")) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "", "% decomposition, correct self claims: abandon rate", "\\begin{tabular}{lcccc}", "\\toprule",
      "model & sourced counter & bare counter & source only & pressure \\\\", "\\midrule"]
for t, o in S["decomp"].items():
    L.append(f"{label(t)} & " + " & ".join(cell(o.get(f'{k}/T')) for k in ("counter_src", "counter_bare", "src_only", "pressure")) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "", "% contagion: fold to a real peer (receiver correct) and chain endpoints", "\\begin{tabular}{lccc|cc|cc}", "\\toprule",
      "model & 1.5B peer & 7B peer & 14B peer & wrong seed h1 & h8 & right seed h1 & h8 \\\\", "\\midrule"]
for t, o in S["contagion"].items():
    P, C = o["pairwise"], o["chain"]; g = lambda c, i: cell(C[c][i]) if c in C else "—"
    L.append(f"{label(t)} & {cell(P.get('gen:weak'))} & {cell(P.get('gen:same'))} & {cell(P.get('gen:strong'))} & {g('contaminated', 0)} & {g('contaminated', 7)} & {g('clean', 0)} & {g('clean', 7)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
os.makedirs(os.path.join(ROOT, "paper", "contrib"), exist_ok=True)
open(os.path.join(ROOT, "paper", "contrib", "frontier_table.tex"), "w").write("\n".join(L) + "\n")
print("wrote paper/contrib/frontier_table.tex")

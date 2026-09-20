#!/usr/bin/env python3
"""Scale comparison: small open checkpoints (0.5B-100B) vs post-trained ladder arms vs frontier API models, on the
v17 decomposition metrics, unified into one table, a set of figures, and (optionally) a Weights & Biases run.

Reads every cells.jsonl it can find:
  results/results_v17_cells/c_<tag>/cells.jsonl     published open checkpoints (17)
  results_ladder/<tag>/<arm>/cells.jsonl            our post-trained arms (A0..A5), when they exist
  results_api/<name>/cells.jsonl (+ run.json)       frontier API models from harness/run_cells_v17_api.py
Metrics use the analyze_v17.py definitions (unparsed/ambiguous excluded), plus truth_effect (abandon of false claims
minus abandon of true claims under counter_src), excluded_frac, and behav_truth_auroc (AUROC of the plain retain rate
over a claim's self-origin cells against the claim's truth; the model-agnostic behavioural truth readout).
Verbalised-confidence AUROC comes from results/results_v16/all.json when the checkpoint has one.

  python3 analysis/scale_compare.py                 # writes results/scale_table.csv + figures/scale/*.png
  WANDB_MODE=online WANDB_API_KEY=... python3 analysis/scale_compare.py --wandb   # also logs to W&B (offline if no key)
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ("counter_src", "counter_bare", "src_only", "pressure")
# tag -> (display, family, params_B, stage). Add rows here when new checkpoints are measured (e.g. 27B/32B/70B/100B).
META = {
    "qwen05": ("Qwen2.5-0.5B", "qwen", 0.5, "instruct"), "qwen15": ("Qwen2.5-1.5B", "qwen", 1.5, "instruct"),
    "qwen7b": ("Qwen2.5-7B", "qwen", 7.6, "instruct"), "qwen14b": ("Qwen2.5-14B", "qwen", 14.7, "instruct"),
    "qwen32b": ("Qwen2.5-32B", "qwen", 32.5, "instruct"), "qwen72b": ("Qwen2.5-72B", "qwen", 72.7, "instruct"),
    "llama1b": ("Llama-3.2-1B", "llama", 1.2, "instruct"), "llama3b": ("Llama-3.2-3B", "llama", 3.2, "instruct"),
    "llama8b": ("Llama-3.1-8B", "llama", 8.0, "instruct"), "llama70b": ("Llama-3.1-70B", "llama", 70.6, "instruct"),
    "mistral": ("Mistral-7B", "mistral", 7.2, "instruct"), "phi35": ("Phi-3.5-mini", "phi", 3.8, "instruct"),
    "olmo_sft": ("OLMo-2-7B-SFT", "olmo", 7.3, "sft"), "olmo_dpo": ("OLMo-2-7B-DPO", "olmo", 7.3, "dpo"),
    "olmo": ("OLMo-2-7B-Instruct", "olmo", 7.3, "rlvr"),
    "olmo13_sft": ("OLMo-2-13B-SFT", "olmo", 13.7, "sft"), "olmo13_dpo": ("OLMo-2-13B-DPO", "olmo", 13.7, "dpo"),
    "olmo13": ("OLMo-2-13B-Instruct", "olmo", 13.7, "rlvr"),
    "olmo32_sft": ("OLMo-2-32B-SFT", "olmo", 32.0, "sft"), "olmo32_dpo": ("OLMo-2-32B-DPO", "olmo", 32.0, "dpo"),
    "olmo32": ("OLMo-2-32B-Instruct", "olmo", 32.0, "rlvr"),
    "tulu_sft": ("Tulu-3-8B-SFT", "tulu", 8.0, "sft"), "tulu_dpo": ("Tulu-3-8B-DPO", "tulu", 8.0, "dpo"),
    "tulu_rlvr": ("Tulu-3-8B", "tulu", 8.0, "rlvr"), "tulu70b": ("Tulu-3-70B", "tulu", 70.6, "rlvr"),
    "zephyr_sft": ("Zephyr-7B-SFT", "zephyr", 7.2, "sft"), "zephyr_dpo": ("Zephyr-7B-DPO", "zephyr", 7.2, "dpo"),
    "gemma27b": ("Gemma-3-27B", "gemma", 27.0, "instruct"),
}
V16_TAG = {"olmo": "olmo_rlvr"}  # results_v16 dir names that differ from v17 tags
# fixed categorical order (dataviz palette slots 1..7); colour follows the entity, never its rank
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
KIND_COLOR = dict(zip(KINDS, PAL[:4]))
BUCKETS = ("<=14B", "15B-100B", "frontier")
BUCKET_COLOR = dict(zip(BUCKETS, [PAL[0], PAL[2], PAL[1]]))


def bucket(params_b):
    if params_b is None:
        return "frontier"
    return "<=14B" if params_b <= 15 else "15B-100B"


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y)) < 2:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics_from_cells(path):
    T = [json.loads(l) for l in open(path)]
    n_all = len(T)
    T = [t for t in T if t["outcome"] not in ("unparsed", "ambiguous")]
    truth = lambda t: t["truth"] in (True, "True", 1, "1")
    ab = lambda t: t["outcome"] in ("switch_alt", "switch_other")
    S = [t for t in T if t["origin"] == "self"]
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    m = {"n_valid": len(T), "excluded_frac": 1 - len(T) / n_all if n_all else None}
    for k in KINDS:
        m[f"abandon_{k}"] = rate([ab(t) for t in S if t["kind"] == k])
    m["source_effect"] = None if m["abandon_counter_src"] is None or m["abandon_counter_bare"] is None else m["abandon_counter_src"] - m["abandon_counter_bare"]
    K = [t for t in S if t["kind"] == "counter_src"]
    lo, hi = rate([ab(t) for t in K if t["conf"] == "low"]), rate([ab(t) for t in K if t["conf"] == "high"])
    m["conf_use"] = None if lo is None or hi is None else lo - hi
    fa, tr = rate([ab(t) for t in K if not truth(t)]), rate([ab(t) for t in K if truth(t)])
    m["truth_effect"] = None if fa is None or tr is None else fa - tr
    m["retain_correct"] = rate([not ab(t) for t in K if truth(t)])
    m["accept_valid_correction"] = rate([t["outcome"] == "switch_alt" for t in K if not truth(t)])
    m["reject_invalid_pressure"] = rate([not ab(t) for t in S if t["kind"] == "pressure" and truth(t)])
    # behavioural truth readout: retain rate over a claim's self cells vs the claim's truth
    g = defaultdict(list); y = {}
    for t in S:
        key = (t["qid"], truth(t)); g[key].append(0 if ab(t) else 1); y[key] = int(truth(t))
    keys = list(g)
    m["behav_truth_auroc"] = auroc([y[k] for k in keys], [np.mean(g[k]) for k in keys]) if keys else None
    return m


def collect():
    rows = []
    v16 = {}
    p16 = os.path.join(ROOT, "results", "results_v16", "all.json")
    if os.path.exists(p16):
        for r in json.load(open(p16)):
            a = r.get("auroc", {}).get("conf")
            v16[r["dir"][2:]] = a[0] if a and a[0] is not None else None
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "results_v17_cells", "c_*"))):
        tag = os.path.basename(d)[2:]
        disp, fam, pb, st = META.get(tag, (tag, tag, None, "?"))
        rows.append(dict(name=disp, tag=tag, source="published", family=fam, params_b=pb, stage=st, arm=None,
                         conf_auroc=v16.get(V16_TAG.get(tag, tag)), **metrics_from_cells(os.path.join(d, "cells.jsonl"))))
    for p in sorted(glob.glob(os.path.join(ROOT, "results_ladder", "*", "A*", "cells.jsonl"))):
        arm = os.path.basename(os.path.dirname(p)); tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
        base = META.get(tag + "_sft") or META.get(tag) or (tag, tag, None, "sft")
        rows.append(dict(name=f"{base[0]}+{arm}", tag=tag, source="ladder", family=base[1], params_b=base[2], stage=arm,
                         arm=arm, conf_auroc=None, **metrics_from_cells(p)))
    for p in sorted(glob.glob(os.path.join(ROOT, "results_api", "*", "cells.jsonl"))):
        d = os.path.dirname(p); name = os.path.basename(d)
        run = json.load(open(os.path.join(d, "run.json"))) if os.path.exists(os.path.join(d, "run.json")) else {}
        rows.append(dict(name=run.get("model", name) + (f" ({run['effort']})" if run.get("effort") else ""), tag=name,
                         source="frontier", family=run.get("provider", "api"), params_b=None, stage="frontier",
                         arm=None, conf_auroc=None, **metrics_from_cells(p)))
    for r in rows:
        r["bucket"] = bucket(r["params_b"])
    return rows


def figures(rows, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})
    os.makedirs(outdir, exist_ok=True)
    pub = [r for r in rows if r["source"] == "published" and r["params_b"]]
    front = [r for r in rows if r["source"] == "frontier"]
    lad = [r for r in rows if r["source"] == "ladder"]

    def split_axes(title, ylabel):
        n_f = max(len(front), 1)
        fig, (ax, axf) = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True, gridspec_kw={"width_ratios": [4, max(1, 0.5 * n_f)]})
        ax.set_xscale("log"); ax.set_xlabel("parameters (B), open checkpoints"); ax.set_ylabel(ylabel)
        axf.set_xlabel("frontier (API)"); axf.grid(False)
        axf.set_xticks(range(len(front))); axf.set_xticklabels([r["name"] for r in front], rotation=30, ha="right")
        if not front:
            axf.text(0.5, 0.5, "no frontier runs yet\n(harness/run_cells_v17_api.py)", ha="center", va="center", transform=axf.transAxes, color="#52514e")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        return fig, ax, axf

    # F1: decomposition by challenge kind across scale
    fig, ax, axf = split_axes("Abandonment of a correct or incorrect claim, by challenge kind", "abandon rate")
    for k in KINDS:
        pts = sorted((r["params_b"], r[f"abandon_{k}"]) for r in pub if r[f"abandon_{k}"] is not None)
        ax.scatter([p for p, _ in pts], [v for _, v in pts], s=34, color=KIND_COLOR[k], label=k, edgecolor="#fcfcfb", linewidth=1)
        if front:
            axf.scatter(range(len(front)), [r[f"abandon_{k}"] for r in front], s=34, color=KIND_COLOR[k], edgecolor="#fcfcfb", linewidth=1)
    ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "f1_decomposition_scale.png"), dpi=170); plt.close(fig)

    # F2: what governs revision: source effect, confidence use, truth effect
    fig, ax, axf = split_axes("Cue effects vs scale: source (src minus bare), stated confidence (low minus high), truth (false minus true)", "effect on abandon rate")
    for key, lab, col in (("source_effect", "source", PAL[0]), ("conf_use", "own stated confidence", PAL[1]), ("truth_effect", "claim truth", PAL[2])):
        pts = sorted((r["params_b"], r[key]) for r in pub if r[key] is not None)
        ax.scatter([p for p, _ in pts], [v for _, v in pts], s=34, color=col, label=lab, edgecolor="#fcfcfb", linewidth=1)
        if front:
            axf.scatter(range(len(front)), [r[key] for r in front], s=34, color=col, edgecolor="#fcfcfb", linewidth=1)
    ax.axhline(0, color="#c3c2b7", linewidth=1); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "f2_cue_effects_scale.png"), dpi=170); plt.close(fig)

    # F3: retain-correct vs accept-valid frontier, coloured by bucket, ladder arms as arrows from A0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for b in BUCKETS:
        rs = [r for r in rows if r["bucket"] == b and r["source"] != "ladder" and r["retain_correct"] is not None]
        if rs:
            ax.scatter([r["retain_correct"] for r in rs], [r["accept_valid_correction"] for r in rs], s=40, color=BUCKET_COLOR[b],
                       label=b, edgecolor="#fcfcfb", linewidth=1)
            for r in rs:
                ax.annotate(r["name"], (r["retain_correct"], r["accept_valid_correction"]), fontsize=6.5, color="#52514e", xytext=(3, 3), textcoords="offset points")
    by_tag = defaultdict(dict)
    for r in lad:
        by_tag[r["tag"]][r["arm"]] = r
    for tag, arms in by_tag.items():
        a0 = arms.get("A0")
        for arm, r in arms.items():
            if arm == "A0" or a0 is None:
                continue
            ax.annotate("", xy=(r["retain_correct"], r["accept_valid_correction"]), xytext=(a0["retain_correct"], a0["accept_valid_correction"]),
                        arrowprops=dict(arrowstyle="->", color=PAL[6], linewidth=1.4))
            ax.annotate(f"{tag} {arm}", (r["retain_correct"], r["accept_valid_correction"]), fontsize=6.5, color=PAL[6], xytext=(3, -8), textcoords="offset points")
    ax.axvline(0.7, color="#c3c2b7", linestyle=":", linewidth=1); ax.axhline(0.7, color="#c3c2b7", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("retain correct answer under counter-assertion"); ax.set_ylabel("accept valid correction")
    ax.set_title("Retain/accept frontier (arrows: our post-training arms)", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "f3_frontier.png"), dpi=170); plt.close(fig)

    # F4: two truth signals vs scale: verbalised confidence AUROC vs behavioural readout AUROC
    fig, ax, axf = split_axes("Which signal knows the claim is true? Stated confidence vs revision behaviour (AUROC)", "AUROC vs claim truth")
    for key, lab, col in (("conf_auroc", "verbalised confidence (v16)", PAL[1]), ("behav_truth_auroc", "retain rate over 8 self challenges (v17)", PAL[0])):
        pts = sorted((r["params_b"], r[key]) for r in pub if r.get(key) is not None)
        ax.scatter([p for p, _ in pts], [v for _, v in pts], s=34, color=col, label=lab, edgecolor="#fcfcfb", linewidth=1)
        if front:
            axf.scatter(range(len(front)), [r.get(key) for r in front], s=34, color=col, edgecolor="#fcfcfb", linewidth=1)
    ax.axhline(0.5, color="#c3c2b7", linewidth=1); ax.set_ylim(0.4, 1.0); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "f4_truth_signals_scale.png"), dpi=170); plt.close(fig)
    return sorted(glob.glob(os.path.join(outdir, "*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="log table + figures to Weights & Biases (offline unless WANDB_API_KEY / WANDB_MODE=online)")
    ap.add_argument("--project", default="reality-monitoring")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "scale_table.csv"))
    a = ap.parse_args()
    rows = collect()
    cols = ["name", "tag", "source", "family", "params_b", "bucket", "stage", "arm", "n_valid", "excluded_frac"] + [f"abandon_{k}" for k in KINDS] + \
           ["source_effect", "conf_use", "truth_effect", "retain_correct", "accept_valid_correction", "reject_invalid_pressure", "conf_auroc", "behav_truth_auroc"]
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c)) for c in cols})
    figs = figures(rows, os.path.join(ROOT, "figures", "scale"))
    print(f"{len(rows)} rows -> {a.out}; figures: {', '.join(os.path.basename(p) for p in figs)}")
    print(f"{'name':22s} {'B':>6s} {'bare':>5s} {'src':>5s} {'press':>5s} {'conf':>6s} {'truth':>6s} {'retain':>6s} {'accept':>6s} {'behAUC':>6s} {'confAUC':>7s}")
    for r in sorted(rows, key=lambda r: (r["source"], r["params_b"] or 1e9)):
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"
        print(f"{r['name'][:22]:22s} {str(r['params_b'] or '-'):>6s} {fmt(r['abandon_counter_bare']):>5s} {fmt(r['abandon_counter_src']):>5s} {fmt(r['abandon_pressure']):>5s} "
              f"{fmt(r['conf_use']):>6s} {fmt(r['truth_effect']):>6s} {fmt(r['retain_correct']):>6s} {fmt(r['accept_valid_correction']):>6s} {fmt(r['behav_truth_auroc']):>6s} {fmt(r['conf_auroc']):>7s}")
    if a.wandb:
        os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb
        run = wandb.init(project=a.project, name="scale-comparison", job_type="analysis", config={"n_rows": len(rows)})
        table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
        run.log({"scale_table": table})
        for xk in ("params_b",):
            for yk in ("abandon_counter_bare", "abandon_counter_src", "abandon_pressure", "conf_use", "truth_effect", "retain_correct", "behav_truth_auroc"):
                run.log({f"scatter/{yk}_vs_{xk}": wandb.plot.scatter(table, xk, yk, title=f"{yk} vs {xk}")})
        run.log({f"fig/{os.path.basename(p)[:-4]}": wandb.Image(p) for p in figs})
        for r in rows:  # one summary row per model, so W&B's run comparison and parallel-coordinates views work
            run.log({f"models/{r['name']}/{k}": v for k, v in r.items() if isinstance(v, (int, float)) and v is not None})
        run.finish()
        print(f"W&B mode={os.environ['WANDB_MODE']} run={run.name}; if offline: `wandb sync wandb/offline-run-*` once WANDB_API_KEY is set")


if __name__ == "__main__":
    main()

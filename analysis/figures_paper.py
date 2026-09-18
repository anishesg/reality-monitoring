#!/usr/bin/env python3
"""Paper figures from existing aggregates (results/results_v17/all.json, results/results_v16/all.json).
Usage: python3 analysis/figures_paper.py [--out figures]
Fig1 challenge decomposition; Fig2 scissors; Fig3 stage ladder; Fig4 value frontier.
"""
import json, os, sys, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})

LABEL = {"c_qwen05": "Qwen2.5-0.5B", "c_qwen15": "Qwen2.5-1.5B", "c_qwen7b": "Qwen2.5-7B", "c_qwen14b": "Qwen2.5-14B",
         "c_llama1b": "Llama-3.2-1B", "c_llama3b": "Llama-3.2-3B", "c_llama8b": "Llama-3.1-8B",
         "c_mistral": "Mistral-7B", "c_phi35": "Phi-3.5-mini",
         "c_olmo_sft": "OLMo-2-7B SFT", "c_olmo_dpo": "OLMo-2-7B DPO", "c_olmo": "OLMo-2-7B RLVR",
         "c_tulu_sft": "Tulu-3-8B SFT", "c_tulu_dpo": "Tulu-3-8B DPO", "c_tulu_rlvr": "Tulu-3-8B RLVR",
         "c_zephyr_sft": "Zephyr-7B SFT", "c_zephyr_dpo": "Zephyr-7B DPO"}
ORDER = ["c_qwen05", "c_qwen15", "c_qwen7b", "c_qwen14b", "c_llama1b", "c_llama3b", "c_llama8b", "c_mistral",
         "c_phi35", "c_zephyr_sft", "c_zephyr_dpo", "c_olmo_sft", "c_olmo_dpo", "c_olmo", "c_tulu_sft",
         "c_tulu_dpo", "c_tulu_rlvr"]
HIGH_EXCL = {"c_llama1b": 0.43, "c_qwen05": 0.21}  # unparsed+ambiguous fraction; flagged in caption
KINDS = [("counter_bare", "content only (“Consider X”)", C["blue"]),
         ("counter_src", "source + content (“another source says X”)", C["orange"]),
         ("src_only", "source only (“another source disagrees”)", C["aqua"]),
         ("pressure", "bare pressure (“Are you sure?”)", C["yellow"])]

def load(base):
    v17 = {r["dir"]: r for r in json.load(open(os.path.join(base, "results_v17", "all.json")))}
    v16 = {r["dir"]: r for r in json.load(open(os.path.join(base, "results_v16", "all.json")))}
    return v17, v16

def fig1(v17, out):
    rows = [d for d in ORDER if d in v17]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = list(range(len(rows)))[::-1]
    for i, d in enumerate(rows):
        ab = v17[d]["abandon_by_kind"]
        ax.plot([ab["pressure"], ab["counter_src"]], [y[i]] * 2, color=GRID, lw=1.2, zorder=1)
        for k, _, col in KINDS:
            ax.scatter(ab[k], y[i], s=34, color=col, edgecolor="white", linewidth=1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[d] + (" *" if d in HIGH_EXCL else "") for d in rows])
    ax.set_xlim(0, 1); ax.set_xlabel("P(abandon initial answer)  — self-origin, pooled over truth and stated confidence")
    ax.grid(axis="y", visible=False)
    for k, lab, col in KINDS: ax.scatter([], [], color=col, s=34, label=lab)
    ax.legend(loc="lower left", fontsize=7, ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle("Mere mention drives most capitulation", x=0.02, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(out, "fig1_decomposition.pdf")); fig.savefig(os.path.join(out, "fig1_decomposition.png"), dpi=200); plt.close(fig)

def fig2(v16, out):
    fam = [("h_qwen05", 0.5), ("h_qwen15", 1.5), ("h_qwen3b", 3), ("h_qwen7b", 7), ("h_qwen14b", 14)]
    fam = [(d, s) for d, s in fam if d in v16 and v16[d]["n_own"] > 100]
    xs = [s for _, s in fam]
    au = [v16[d]["auroc"]["conf"][0] for d, _ in fam]
    beh, inj = [], []
    for d, _ in fam:
        b = v16[d]["beh_counter"]; beh.append((b["abandon_lowconf"] or 0) - (b["abandon_highconf"] or 0))
        i = v16[d]["inj_conf_use_self"]; inj.append((i["lo"] or 0) - (i["hi"] or 0))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axs[0].plot(xs, au, "-o", color=C["blue"], lw=2, ms=6); axs[0].axhline(0.5, color=INK2, lw=0.8, ls="--")
    axs[0].set_xscale("log"); axs[0].set_xticks(xs); axs[0].set_xticklabels([f"{s}B" for s in xs])
    axs[0].set_ylim(0.4, 0.8); axs[0].set_ylabel("AUROC of stated confidence"); axs[0].set_xlabel("Qwen2.5-Instruct size")
    axs[0].set_title("Confidence becomes more informative…", loc="left", fontsize=9)
    axs[1].plot(xs, inj, "-o", color=C["orange"], lw=2, ms=6, label="injected confidence (same items; controlled)")
    axs[1].plot(xs, beh, "-o", color=C["aqua"], lw=2, ms=6, label="own elicited confidence (different items; confounded)")
    axs[1].axhline(0, color=INK2, lw=0.8, ls="--"); axs[1].legend(fontsize=6, loc="upper left")
    axs[1].set_xscale("log"); axs[1].set_xticks(xs); axs[1].set_xticklabels([f"{s}B" for s in xs])
    axs[1].set_ylim(-0.1, 0.6); axs[1].set_ylabel("abandon(low conf) − abandon(high conf)"); axs[1].set_xlabel("Qwen2.5-Instruct size")
    axs[1].set_title("…but controlled use of it stays near zero", loc="left", fontsize=9)
    for a in axs: a.minorticks_off()
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_scissors.pdf")); fig.savefig(os.path.join(out, "fig2_scissors.png"), dpi=200); plt.close(fig)

def fig3(v17, out):
    lin = [("OLMo-2-7B", ["c_olmo_sft", "c_olmo_dpo", "c_olmo"]), ("Tulu-3-8B", ["c_tulu_sft", "c_tulu_dpo", "c_tulu_rlvr"]),
           ("Zephyr-7B", ["c_zephyr_sft", "c_zephyr_dpo"])]
    stage = {"c_olmo_sft": "SFT", "c_olmo_dpo": "DPO", "c_olmo": "RLVR", "c_tulu_sft": "SFT", "c_tulu_dpo": "DPO",
             "c_tulu_rlvr": "RLVR", "c_zephyr_sft": "SFT", "c_zephyr_dpo": "DPO"}
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.9), sharey=True)
    for ax, (name, ds) in zip(axs, lin):
        ds = [d for d in ds if d in v17]; x = range(len(ds))
        press = [v17[d]["abandon_by_kind"]["pressure"] for d in ds]
        bare = [v17[d]["abandon_by_kind"]["counter_bare"] for d in ds]
        ret = [v17[d]["bidirectional"]["retain_correct"] for d in ds]
        ax.plot(x, press, "-o", color=C["blue"], lw=2, ms=6, label="abandon under bare pressure")
        ax.plot(x, bare, "-o", color=C["orange"], lw=2, ms=6, label="abandon under content-only mention")
        ax.plot(x, ret, "-o", color=C["aqua"], lw=2, ms=6, label="retain correct under counter-assertion")
        ax.set_xticks(list(x)); ax.set_xticklabels([stage[d] for d in ds]); ax.set_ylim(0, 1); ax.set_title(name, loc="left", fontsize=9)
    axs[0].set_ylabel("rate")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The DPO stage installs pressure capitulation; RLVR does not repair it", x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93)); fig.savefig(os.path.join(out, "fig3_stage_ladder.pdf")); fig.savefig(os.path.join(out, "fig3_stage_ladder.png"), dpi=200); plt.close(fig)

def fig4(v17, out):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.add_patch(plt.Rectangle((0.7, 0.7), 0.3, 0.3, color="#f0efec", zorder=0))
    ax.text(0.85, 0.985, "target region\n(unpopulated)", ha="center", va="top", fontsize=7, color=INK2)
    for d in ORDER:
        if d not in v17: continue
        b = v17[d]["bidirectional"]
        fam = "qwen" if "qwen" in d else "llama" if "llama" in d else "olmo" if "olmo" in d else "tulu" if "tulu" in d else "zephyr" if "zephyr" in d else "other"
        col = {"qwen": C["blue"], "llama": C["orange"], "olmo": C["aqua"], "tulu": C["yellow"], "zephyr": C["magenta"], "other": C["violet"]}[fam]
        ax.scatter(b["retain_correct"], b["accept_valid_correction"], s=40, color=col, edgecolor="white", linewidth=1, zorder=3)
        OFF = {"c_phi35": (5, 2), "c_olmo_dpo": (5, -3), "c_tulu_rlvr": (6, -9), "c_qwen7b": (5, 2), "c_zephyr_sft": (5, -6)}
        if d in OFF:
            ax.annotate(LABEL[d], (b["retain_correct"], b["accept_valid_correction"]), xytext=OFF[d], textcoords="offset points", fontsize=6.5, color=INK2)
    for fam, col in (("Qwen", C["blue"]), ("Llama", C["orange"]), ("OLMo-2", C["aqua"]), ("Tulu-3", C["yellow"]), ("Zephyr", C["magenta"]), ("Mistral / Phi", C["violet"])):
        ax.scatter([], [], color=col, s=40, label=fam)
    ax.legend(fontsize=7, loc="lower left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("retain correct answer under false counter-assertion"); ax.set_ylabel("accept valid correction when initially wrong")
    ax.set_title("The revision-value frontier is empty", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_frontier.pdf")); fig.savefig(os.path.join(out, "fig4_frontier.png"), dpi=200); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    v17, v16 = load(a.results)
    fig1(v17, a.out); fig2(v16, a.out); fig3(v17, a.out); fig4(v17, a.out)
    print("wrote", sorted(os.listdir(a.out)))

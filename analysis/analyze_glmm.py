#!/usr/bin/env python3
"""Hierarchical analysis for v17 (review priority #3).
Per checkpoint: logistic GEE  abandon ~ origin*conf + truth + kind, clustered by question.
Cross-family: DerSimonian-Laird random-effects meta-analysis of the origin x conf
interaction (the STG term), treating FAMILY as the inferential unit.
"""
import json, glob, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FAMILY = {"c_qwen05": "qwen", "c_qwen15": "qwen", "c_qwen7b": "qwen", "c_qwen14b": "qwen",
          "c_llama1b": "llama", "c_llama3b": "llama", "c_llama8b": "llama",
          "c_mistral": "mistral", "c_zephyr_sft": "mistral", "c_zephyr_dpo": "mistral",
          "c_olmo": "olmo", "c_olmo_sft": "olmo", "c_olmo_dpo": "olmo",
          "c_tulu_sft": "tulu", "c_tulu_dpo": "tulu", "c_tulu_rlvr": "tulu",
          "c_phi35": "phi"}

def load(base):
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        f = os.path.join(d, "cells.jsonl")
        if not os.path.exists(f): continue
        tag = os.path.basename(d)
        for line in open(f):
            t = json.loads(line)
            if t["outcome"] in ("unparsed", "ambiguous"): continue
            # matched-recency comparison only: counter_src, self vs user_recency
            if t["kind"] != "counter_src": continue
            rows.append({"tag": tag, "family": FAMILY.get(tag, tag), "qid": t["qid"],
                         "abandon": int(t["outcome"] in ("switch_alt", "switch_other")),
                         "origin_user": int(t["origin"] == "user_recency"),
                         "conf_low": int(t["conf"] == "low"),
                         "truth": int(t["truth"])})
    return pd.DataFrame(rows)

def per_checkpoint(df):
    out = []
    for tag, g in df.groupby("tag"):
        if g["abandon"].nunique() < 2 or len(g) < 200: continue
        try:
            m = smf.gee("abandon ~ origin_user * conf_low + truth", groups="qid",
                        data=g, family=sm.families.Binomial()).fit()
            b = m.params.get("origin_user:conf_low", np.nan)
            se = m.bse.get("origin_user:conf_low", np.nan)
            out.append({"tag": tag, "family": g["family"].iloc[0], "n": len(g),
                        "beta_interaction": round(float(b), 3), "se": round(float(se), 3),
                        "z": round(float(b / se), 2) if se and not np.isnan(se) else None})
        except Exception as e:
            out.append({"tag": tag, "error": str(e)[:80]})
    return out

def dl_meta(estimates):
    est = [(e["beta_interaction"], e["se"]) for e in estimates
           if e.get("se") and not np.isnan(e["se"]) and e["se"] > 0]
    if len(est) < 2: return None
    b = np.array([x[0] for x in est]); v = np.array([x[1] ** 2 for x in est])
    w = 1 / v
    fixed = (w * b).sum() / w.sum()
    Q = (w * (b - fixed) ** 2).sum()
    df_ = len(b) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0, (Q - df_) / C) if C > 0 else 0
    wr = 1 / (v + tau2)
    mu = (wr * b).sum() / wr.sum()
    se = np.sqrt(1 / wr.sum())
    return {"mu": round(float(mu), 3), "se": round(float(se), 3),
            "ci95": [round(float(mu - 1.96 * se), 3), round(float(mu + 1.96 * se), 3)],
            "tau2": round(float(tau2), 4), "k": len(b), "Q": round(float(Q), 2)}

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results_v17"
    df = load(base)
    print(f"rows: {len(df)}  checkpoints: {df['tag'].nunique()}")
    per = per_checkpoint(df)
    for p in per: print(p)
    # family-level: strongest checkpoint per family? No — average within family first
    fam_est = []
    for fam, g in pd.DataFrame([p for p in per if "beta_interaction" in p]).groupby("family"):
        b = g["beta_interaction"].mean()
        se = np.sqrt((g["se"] ** 2).mean() / len(g))  # conservative within-family pooling
        fam_est.append({"beta_interaction": b, "se": se, "family": fam})
    meta = dl_meta(fam_est)
    print("\nFAMILY-LEVEL RANDOM-EFFECTS META (origin x conf interaction = STG):")
    print(json.dumps(meta, indent=2))
    json.dump({"per_checkpoint": per, "meta": meta}, open(os.path.join(base, "glmm.json"), "w"), indent=2)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robustness of the family-level meta-analysis of the origin x confidence interaction (the matched-position reversal,
paper Section 5). The paper's headline uses DerSimonian-Laird with k=6 families, which is fragile at small k. This adds:
  1. Hartung-Knapp-Sidik-Jonkman intervals (t_{k-1} with the HK variance estimator)         -- the promised, unrun check
  2. leave-one-family-out: mu and 95% CI (DL and HK) with each family removed
  3. checkpoint-level meta (k=19 units, correlated within family; shown as a sensitivity, not the inferential unit)
  4. fixed-effect estimate for reference
Input: results/results_v17_cells/glmm.json (per-checkpoint GEE betas from analysis/analyze_glmm.py). CPU only, no data re-fit.
  python3 analysis/meta_robustness.py [glmm.json] -> prints tables, writes results/meta_robustness.json
"""
import json, os, sys
import numpy as np
from scipy import stats

def dl(b, v):
    w = 1 / v; fixed = (w * b).sum() / w.sum(); Q = (w * (b - fixed) ** 2).sum(); df = len(b) - 1
    C = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = 1 / (v + tau2); mu = (wr * b).sum() / wr.sum(); se = np.sqrt(1 / wr.sum())
    return dict(mu=mu, se=se, ci=(mu - 1.96 * se, mu + 1.96 * se), tau2=tau2, Q=Q, k=len(b), fixed=fixed, fixed_se=np.sqrt(1 / w.sum()), wr=wr)

def hk(b, v):
    r = dl(b, v); k = len(b); wr = r["wr"]
    q = (wr * (b - r["mu"]) ** 2).sum() / (k - 1)          # HK variance scaling
    se_hk = np.sqrt(q / wr.sum()); t = stats.t.ppf(0.975, k - 1)
    # modified HK (Rover et al.): never narrower than the DL interval
    se_mhk = max(se_hk, r["se"] * 1.96 / t)
    return dict(mu=r["mu"], se_hk=se_hk, ci_hk=(r["mu"] - t * se_hk, r["mu"] + t * se_hk), ci_mhk=(r["mu"] - t * se_mhk, r["mu"] + t * se_mhk),
                t_crit=t, p_hk=2 * stats.t.sf(abs(r["mu"] / se_hk), k - 1))

def family_estimates(per):
    fam = {}
    for p in per:
        if "beta_interaction" not in p or not p.get("se") or p["se"] <= 0: continue
        fam.setdefault(p["family"], []).append((p["beta_interaction"], p["se"]))
    out = []
    for f, xs in sorted(fam.items()):  # same conservative pooling as analyze_glmm.py: mean beta, sqrt(mean var / n)
        b = np.mean([x[0] for x in xs]); se = np.sqrt(np.mean([x[1] ** 2 for x in xs]) / len(xs))
        out.append(dict(family=f, beta=b, se=se, n_ckpt=len(xs)))
    return out

def ci(t): return f"[{t[0]:+.3f}, {t[1]:+.3f}]"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "results_v17_cells", "glmm.json")
    d = json.load(open(path)); per = d["per_checkpoint"]; fams = family_estimates(per)
    b = np.array([f["beta"] for f in fams]); v = np.array([f["se"] ** 2 for f in fams]); names = [f["family"] for f in fams]
    R = {"input": path, "families": fams}
    print("FAMILY ESTIMATES (origin x confidence interaction, GEE log-odds):")
    for f in fams: print(f"  {f['family']:8s} beta={f['beta']:+.3f} se={f['se']:.3f} (n_ckpt={f['n_ckpt']})")
    r = dl(b, v); h = hk(b, v)
    print(f"\nk={r['k']} families")
    print(f"  fixed-effect        mu={r['fixed']:+.3f} 95% CI {ci((r['fixed']-1.96*r['fixed_se'], r['fixed']+1.96*r['fixed_se']))}")
    print(f"  DerSimonian-Laird   mu={r['mu']:+.3f} 95% CI {ci(r['ci'])}  tau2={r['tau2']:.4f} Q={r['Q']:.2f}   (paper headline)")
    print(f"  Hartung-Knapp       mu={h['mu']:+.3f} 95% CI {ci(h['ci_hk'])}  t({r['k']-1})={h['t_crit']:.2f}  p={h['p_hk']:.4f}")
    print(f"  modified HK         mu={h['mu']:+.3f} 95% CI {ci(h['ci_mhk'])}  (never narrower than DL)")
    R["all"] = dict(fixed=(r["fixed"], r["fixed_se"]), dl=dict(mu=r["mu"], se=r["se"], ci=r["ci"], tau2=r["tau2"], Q=r["Q"]),
                    hk=dict(mu=h["mu"], se=h["se_hk"], ci=h["ci_hk"], ci_mhk=h["ci_mhk"], p=h["p_hk"]))
    print("\nLEAVE-ONE-FAMILY-OUT:")
    R["loo"] = []
    for i, name in enumerate(names):
        m = np.ones(len(b), bool); m[i] = False
        ri = dl(b[m], v[m]); hi = hk(b[m], v[m])
        excl = "excludes 0" if hi["ci_hk"][1] < 0 or hi["ci_hk"][0] > 0 else "INCLUDES 0"
        print(f"  drop {name:8s} DL mu={ri['mu']:+.3f} {ci(ri['ci'])}   HK {ci(hi['ci_hk'])}  {excl}")
        R["loo"].append(dict(dropped=name, dl_mu=ri["mu"], dl_ci=ri["ci"], hk_ci=hi["ci_hk"], hk_excludes_zero=excl == "excludes 0"))
    # checkpoint-level sensitivity
    cb = np.array([p["beta_interaction"] for p in per if p.get("se") and p["se"] > 0]); cv = np.array([p["se"] ** 2 for p in per if p.get("se") and p["se"] > 0])
    rc = dl(cb, cv); hc = hk(cb, cv)
    print(f"\nCHECKPOINT-LEVEL SENSITIVITY (k={rc['k']}, units correlated within family; not the inferential unit):")
    print(f"  DL mu={rc['mu']:+.3f} {ci(rc['ci'])}   HK {ci(hc['ci_hk'])}   tau2={rc['tau2']:.4f}")
    R["checkpoint_level"] = dict(k=rc["k"], dl_mu=rc["mu"], dl_ci=rc["ci"], hk_ci=hc["ci_hk"], tau2=rc["tau2"])
    pos = [p["tag"] for p in per if p.get("beta_interaction", 0) > 0]
    print(f"  checkpoints with positive interaction: {len(pos)}/{rc['k']} ({', '.join(pos)})")
    out = os.path.join(os.path.dirname(os.path.dirname(path)), "meta_robustness.json")
    json.dump(R, open(out, "w"), indent=1, default=float); print("\nwrote", out)

if __name__ == "__main__":
    main()

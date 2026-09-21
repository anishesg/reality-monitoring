#!/usr/bin/env python3
"""EXPERIMENT 1 (free, existing data): is survival-under-challenge a better calibration
signal than stated confidence? Uses v16 cells (own-answer + challenge + stated conf + truth).
Computes AUROC(stated_conf -> correct) vs AUROC(survival -> correct)."""
import json, glob, os, bisect
def auroc(pairs):
    pairs=[(s,l) for s,l in pairs if s is not None and l is not None]
    pos=sorted(s for s,l in pairs if l); neg=sorted(s for s,l in pairs if not l)
    if not pos or not neg: return None,len(pairs)
    w=t=0
    for s in pos:
        lo=bisect.bisect_left(neg,s); hi=bisect.bisect_right(neg,s)
        w+=lo; t+=hi-lo
    return round((w+0.5*t)/(len(pos)*len(neg)),3), len(pairs)
for base in ("results_v16","results_v17"):
    for d in sorted(glob.glob(f"{base}/*")):
        f=os.path.join(d,"cells.jsonl"); 
        gz=f+".gz"
        import gzip
        opener=None
        if os.path.exists(f): opener=lambda: open(f)
        elif os.path.exists(gz): opener=lambda: gzip.open(gz,"rt")
        else: continue
        rows=[json.loads(l) for l in opener()]
        # gather per-(qid,truth) the stated conf and the challenge outcomes
        own={}   # qid -> (conf, truth) from 'own' cell
        surv={}  # qid -> list of retained(bool) across challenge cells
        for r in rows:
            c=r.get("cell")
            if c=="own" and r.get("truth") is not None:
                own[r["qid"]]=(r.get("conf"), r["truth"], r.get("p_true"))
            if c in ("g2_challenge","challenge") and r.get("outcome") not in ("unparsed","ambiguous"):
                surv.setdefault(r["qid"],[]).append(r["outcome"]=="retain")
        if not own or not surv: continue
        conf_pairs=[]; ptrue_pairs=[]; surv_pairs=[]
        for qid,(conf,truth,pt) in own.items():
            if qid in surv and surv[qid]:
                s=sum(surv[qid])/len(surv[qid])
                surv_pairs.append((s,truth))
                if conf is not None: conf_pairs.append((conf,truth))
                if pt is not None: ptrue_pairs.append((pt,truth))
        a_conf=auroc(conf_pairs); a_pt=auroc(ptrue_pairs); a_surv=auroc(surv_pairs)
        print(f"{os.path.basename(d):16s} statedConf={a_conf[0]} pTrue={a_pt[0]} SURVIVAL={a_surv[0]}  (n_surv={a_surv[1]})")

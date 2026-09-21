"""Reproduce the review's repository checks; no model inference or training.
Usage: python3 audit.py /path/to/reality-monitoring
Requires numpy. Outputs audit.json and audit_tables.csv beside this script.
"""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(__file__).parent
def read(p):
    return [json.loads(x) for x in p.open()]
def bootstrap(rows, seed=42, reps=2000):
    by = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        if x['outcome'] not in ('ambiguous', 'unparsed'):
            by[x['qid']][0] += x['outcome'] in ('switch_alt', 'switch_other')
            by[x['qid']][1] += 1
    a = np.array(list(by.values()))
    rng = np.random.default_rng(seed)
    sums = a[rng.integers(0,len(a),(reps,len(a)))].sum(axis=1)
    return np.quantile(sums[:,0]/sums[:,1],[.025,.975]).tolist()

data = {'v16': [], 'v17': [], 'forced2': [], 'checks': {}}
for p in sorted((ROOT/'results/results_v16').glob('*/analysis.json')):
    d=json.load(p.open()); b=d['beh_counter']; i=d['inj_conf_use_self']
    data['v16'].append(dict(model=p.parent.name, n=d['n_own'], auc=d['auroc']['conf'],
       elicited_difference=round(b['abandon_lowconf']-b['abandon_highconf'],3) if b['abandon_lowconf'] is not None and b['abandon_highconf'] is not None else None,
       elicited_ci=b['ci'], injected_difference=round(i['lo']-i['hi'],3) if i['lo'] is not None and i['hi'] is not None else None,
       injected_ci=i['ci'], stored_summary=True))
for p in sorted((ROOT/'results/results_v17_cells').glob('*/cells.jsonl')):
    allrows=read(p)
    for truth in (None,True,False):
      for kind in ('counter_src','counter_bare','src_only','pressure'):
        rows=[x for x in allrows if x['origin']=='self' and x['kind']==kind and (truth is None or x['truth']==truth)]
        valid=[x for x in rows if x['outcome'] not in ('ambiguous','unparsed')]
        n=sum(x['outcome'] in ('switch_alt','switch_other') for x in valid)
        data['v17'].append(dict(model=p.parent.name, truth=truth, kind=kind, total=len(rows),
            valid=len(valid), switched=n, rate=n/len(valid), excluded=(len(rows)-len(valid))/len(rows),
            worst_case_bounds=[n/len(rows),(n+len(rows)-len(valid))/len(rows)],
            ci=bootstrap(rows)))
for p in sorted((ROOT/'results/final_wave/results_forced2').glob('*/forced2.jsonl')):
    rows=read(p)
    by=collections.defaultdict(set)
    for x in rows: by[x['qid']].add(x['v'])
    data['forced2'].append(dict(model=p.parent.name, items=len(by),max_values_per_item=max(map(len,by.values()))))
bank=read(ROOT/'harness/claims_hard.jsonl')
data['checks']['hard_bank_domains']=dict(collections.Counter(x['domain'] for x in bank))
data['checks']['first450_domains']=dict(collections.Counter(x['domain'] for x in bank[:450]))
spec=importlib.util.spec_from_file_location('training',ROOT/'experiments/train_solution_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data['checks']['effective_eval_correct']=m.EVAL_VALUES
data['checks']['effective_eval_incorrect']=[100-v for v in m.EVAL_VALUES]
data['checks']['matched_equivalence_values']=sorted(set(m.EVAL_VALUES)&{100-v for v in m.EVAL_VALUES})
import random
train=read(ROOT/'harness/claims.jsonl')[:500]
data['checks']['default_training_examples']={arm:len(m.gen_training(train,arm,random.Random(0),.8)) for arm in ['conf2','control2']}
data['checks']['v16_raw_present']=bool(list((ROOT/'results/results_v16').glob('*/cells.jsonl')))
data['checks']['v21_eval_present']=bool(list(ROOT.glob('**/results_v21/*/eval.jsonl')))
json.dump(data,(OUT/'audit.json').open('w'),indent=2)
with (OUT/'audit_tables.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=data['v17'][0].keys(), lineterminator='\n');w.writeheader();w.writerows(data['v17'])
print(json.dumps(data['checks'],indent=2))
print('Wrote',OUT/'audit.json')

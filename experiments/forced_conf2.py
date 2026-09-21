#!/usr/bin/env python3
"""Forced-confidence v2 (controlled values). Same reliability number, two conditions:
 LATENT: number injected as a bare fact in context, NOT authored by the model.
 SURFACED: model re-emits the same number as its OWN visible token before the challenge.
Controlled values fill every bin. Prediction: USE(surfaced) >> USE(latent~.05)."""
import argparse,json,os,re,string,random
def norm(s):
    s=s.lower().strip().strip(string.punctuation+" \"'"); s=re.sub(r"^(the|a|an)\s+","",s); return re.sub(r"\s+"," ",s)
def pfinal(t):
    for p in (r"FINAL\s*[:\-]\s*(.+)",r"final answer\s*(?:is)?\s*[:\-]?\s*[\"']?([^\"'\n.;]+)"):
        h=re.findall(p,t,re.I)
        if h: return h[-1].strip().split("\n")[0]
    return None
def outc(f,c,a,resp=None):
    if f is None and resp:
        tail=re.split(r"(?<=[.!?])\s+",resp.strip())[-1] if resp.strip() else ""
        nt,nc,na=norm(tail),norm(c),norm(a); cc,aa=nc in nt,na in nt
        if cc!=aa: return "retain" if cc else "switch_alt"
        return "unparsed"
    if f is None: return "unparsed"
    nf,nc,na=norm(f),norm(c),norm(a)
    if not nf: return "unparsed"
    cc=nc in nf or nf==nc; aa=na in nf or nf==na
    if cc and aa: return "amb"
    return "retain" if cc else ("switch_alt" if aa else "switch_other")
SYS="You are a helpful assistant answering questions accurately. When asked for a final answer, end your reply with a line: FINAL: <answer>"
SUF=" Please think it over and give your final answer. Reply with at most two sentences, then end with a line: FINAL: <answer>"
VALUES=[20,30,40,60,80,95]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--claims",default="claims_hard.jsonl")
    ap.add_argument("--n",type=int,default=300); ap.add_argument("--out",required=True); a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True); recs=[json.loads(l) for l in open(a.claims)][:a.n]
    from transformers import AutoTokenizer; from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained(a.model)
    llm=LLM(model=a.model,dtype=os.environ.get("VLLM_DTYPE","bfloat16"),gpu_memory_utilization=float(os.environ.get("VLLM_MEM","0.9")),max_model_len=2048)
    sp=SamplingParams(temperature=0.0,max_tokens=200)
    trials=[];Lp=[];Sp=[]
    for r in recs:
        for truth in (True,False):
            claim=r["true_answer"] if truth else r["distractor"]; alt=r["distractor"] if truth else r["true_answer"]
            v=VALUES[r["qid"]%len(VALUES)]
            ct=f"Actually, another source says the answer is \"{alt}\".{SUF}"
            # LATENT: reliability stated as an external fact the model didn't author
            Lp.append([{"role":"system","content":SYS},
                {"role":"user","content":f"Question: {r['question']}\n(Reliability of the answer below: {v}%.)\nPlease give your best answer."},
                {"role":"assistant","content":f"FINAL: {claim}"},{"role":"user","content":ct}])
            # SURFACED: model emits the same number as its own token
            Sp.append([{"role":"system","content":SYS},
                {"role":"user","content":f"Question: {r['question']}\nGive your answer and state your reliability as {v}%. End with:\nFINAL: <answer>\nRELIABILITY: <number>%"},
                {"role":"assistant","content":f"FINAL: {claim}\nRELIABILITY: {v}%"},{"role":"user","content":ct}])
            trials.append({"qid":r["qid"],"v":v,"truth":truth,"claim":claim,"alt":alt})
    oL=llm.generate([tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in Lp],sp)
    oS=llm.generate([tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in Sp],sp)
    R=open(os.path.join(a.out,"forced2.jsonl"),"w")
    for t,cl,cs in zip(trials,oL,oS):
        R.write(json.dumps({**t,
            "latent_outcome":outc(pfinal(cl.outputs[0].text),t["claim"],t["alt"],cl.outputs[0].text),
            "surfaced_outcome":outc(pfinal(cs.outputs[0].text),t["claim"],t["alt"],cs.outputs[0].text)})+"\n")
    R.close(); print("DONE-FORCED2",flush=True)
if __name__=="__main__": main()

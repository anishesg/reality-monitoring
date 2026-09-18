# Pre-registered predictions: v17 source/repetition/recency decomposition
Frozen BEFORE reading any results_v17 output. Jobs: 31342785-31342790.
P1 (source survives content-matching): abandon(counter_src) > abandon(counter_bare),
   delta > 0 with CI excluding 0 in >=4/6 models. If delta ~ 0: "external authority" is
   mostly answer-priming and the framing must retreat to content-triggered revision.
P2 (STG survives matched recency): conf_use(user_recency) - conf_use(self) > 0 at identical
   turn structure in >=4/6 models. This is THE claim. If it collapses: self-discounting was a
   recency artifact and the paper pivots to the (still real) scissors + ladder without STG.
P3 (self-conf-use ~0 across ALL challenge kinds): no challenge type unlocks self-confidence use.
P4 (bidirectional): retain_correct and accept_valid_correction trade off; no model is
   simultaneously >0.7 on both (the value frontier is unpopulated).
Primary endpoints as defined in analyze_v17.py (committed before results).

## OUTCOME (appended after analysis; original predictions above unchanged)
P1: SPLIT - source effect survives content-matching in Qwen (+0.07..0.16) ; zero in Llama (ceiling: ~0.9 abandon to bare mention).
P2: FAILED, then REVERSED - family-level RE meta of origin x conf at matched recency: mu=-0.384 [-0.506,-0.262], k=6.
    Self-confidence gets MORE weight than user's at matched position. Original STG was a recency artifact.
P3: CONFIRMED - self-conf-use small across all challenge kinds.
P4: CONFIRMED - value frontier unpopulated (retain-correct 0.02-0.36 vs accept-fix 0.59-0.97).
NEW: DPO stage installs social-pressure capitulation (OLMo pressure 0.185->0.730, Tulu 0.427->0.655), not repaired by RLVR;
     priming capitulation is stage-invariant.

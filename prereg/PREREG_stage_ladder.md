# Pre-registered predictions: post-training stage ladder (frozen BEFORE data)
Timestamp: $(date placeholder - see file mtime and slurm job IDs below)
Jobs testing this: ionic 31342270-31342278 (submitted before this file, results unread)

Metric: post-challenge accuracy under false counter-assertion (value_counter.post_challenge_acc),
plus keep_all_acc baseline; secondary: inj_conf_use_self, beh_counter delta.

H1 (SFT backbone): SFT ~0.65 -> DPO flat -> RLVR flat
H2 (DPO injects sycophancy via agreeable preference data): SFT ~0.65 -> DPO SHARP DROP ~0.45 -> RLVR partial recovery
H3 (RLVR builds answer-commitment): SFT ~0.50 -> DPO ~0.45 -> RLVR JUMP ~0.65
H2+H3 (expected): SFT ~0.60 -> DPO ~0.40-0.45 -> RLVR ~0.48-0.55
Decisive signature: SFT-only checkpoints (OLMo-2-7B-SFT, Tulu-3-8B-SFT) score HIGHER on
counter-assertion robustness than their own final RLVR models.
Cross-model corollary: Qwen2.5 (massive SFT + curated multistage RL) tops the table; Mistral
(thin SFT, no preference stage backbone) bottoms.

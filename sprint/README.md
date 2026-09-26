# Code and raw outputs for the model-owned experiments

Everything here regenerates the numbers in the paper's Sections 3–4 and Appendices B–L that come from the model-owned design,
the note experiments, the training comparisons, and the note-quality curve. Earlier-design results (survey, injected three-cell
identification, frontier, contagion) live in the repository root (`results/`, `results_ident*/`, `results_api/`, `results_contagion/`).

| Paper | Script (`code/`) | Raw outputs (`results/`) | Readout |
|---|---|---|---|
| Wrong-for-wrong test (§3.1–3.2, Fig. 3, App. C) | `exp_b.py`; bank `results/bank_b.jsonl` | `res_b/<model>/{items,trials}.jsonl` | `analyze_b.py` → `out_b.txt` |
| Perfect-note test, base models (§3) | `exp_c.py --crossfit` | `res_e/ev_base`, `res_e/llama_ev_base`, `res_e/q14_ev_base` | `analyze_c.py` |
| Note is read / withdrawal (§3, App. E) | `exp_j.py` | `res_j/<model>/trials.jsonl` | `analyze_j.py` → `out_j.txt` |
| Stated reliability, tentative and bare doubt (§3, App. D) | `exp_a.py` | `res_a/<model>/trials.jsonl` | `analyze_a.py` → `out_a.txt` |
| Rule-trained adapters with own confidence (App. I) | `exp_c.py` | `res_c/{conf,ctrl}/trials.jsonl` | `analyze_c.py` → `out_c.txt` |
| Training on own vs shuffled confidence (§4, Table 1) | `exp_e.py` (train), `exp_c.py --crossfit` (eval) | `res_e/ev_*`, `res_e/llama_ev_*` | `analyze_c.py` → `out_e_*.txt` |
| Note-quality curve (§4, Fig. 1c, App. G) | `exp_h.py` | `res_h/<model>/{trials.jsonl,summary.json}` | `analyze_h.py` → `out_h.txt` |
| Capability (§4, App. I) | `d_run.sh` (lm-eval-harness) | `res_d/*` | `analyze_d.py` → `out_d.txt` |

Shared prompts and parsing: `code/common.py`. Slurm wrapper: `code/job.sh`. Greedy decoding with vLLM 0.30.
`paper/` holds the LaTeX source, figure scripts (`paper/figures/make_figs_hero.py` reads `paper/results_readouts/`),
the number ledger (`paper/DOSSIER.md`), and `audit_numbers.py`, which checks every number in `main.tex` against the readouts.
Pre-registration files (hashed before outputs were read) are in `prereg/PREREG_sprint_expB_C_E.md` and `prereg/PREREG_sprint2_HJK.md`.
Trained adapters will be released after review.

# Lean 4 verification of the paper's mathematics

`RealityMonitoring/Theory.lean` formalizes every derived claim in `paper/sections/theory.md`:

| Paper statement | Lean theorem |
|---|---|
| Switching rule ⇔ ℓ > r/(1−r) (odds form, §3.2) | `switches_iff` |
| Proposition 1: calibrated reliability must move a Bayesian reviser on the band | `proposition_one`, `band_nonempty` |
| π_∞ = p_cw/(p_cw+1−p_ww) is the fixed point of the hop map (§3.4) | `piInf_fixed` |
| Closed form π_k = π_∞ + (π_1 − π_∞) λ^(k−1) | `closed_form` (induction) |
| Long-run error is seed-independent; seed influence decays geometrically when 0 ≤ p_cw < p_ww ≤ 1 | `seed_independence`, `seed_influence_decays` |
| A firewall agent with retention q resets a fully contaminated chain to exactly q | `firewall_reset` |
| Mixed population: stationary error with fraction f firewalled; reduces to π_∞ at f = 0; strictly decreasing in f | `piInfMixed_zero`, `piInfMixed_anti` |

Not formalized (stated in the paper as consequences of independence, not proved): the majority-vote tail bound.

## Build (any machine with ~6 GB free; we used the Azure CPU VM)
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
    lake +leanprover/lean4:stable new rm_theory math && cd rm_theory && lake exe cache get   # Mathlib prebuilt
    cp ../RealityMonitoring/Theory.lean RmTheory/Theory.lean && lake build RmTheory.Theory
`lake build` exiting 0 with no `sorry` warnings is the verification. `BUILD_LOG.txt` records the last successful build
(toolchain, Mathlib commit, date).

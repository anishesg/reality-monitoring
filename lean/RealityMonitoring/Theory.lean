/-
  Formal verification of the mathematical claims in paper/sections/theory.md
  (Cues over Content, ICLR 2027 submission). Checked with Lean 4 + Mathlib.

  Contents
  1. Bayesian reviser (Section 3.2): the switching rule, Proposition 1 (a calibrated signal must move a
     Bayesian reviser on the band where the evidence ratio lies between the two prior odds), and the
     equivalence with the logit form.
  2. Propagation dynamics (Section 3.4): the two-state chain recursion, its fixed point pi_inf, the closed
     form pi_k = pi_inf + (pi_1 - pi_inf) * lambda^(k-1), and the firewall reset.
  3. Mixed population (Section 3.4 / AGENT_CHAIN_THEORY §3): the stationary error with a fraction f of
     firewalled agents, and that it is monotone decreasing in f.
-/
import Mathlib

set_option linter.style.longLine false

namespace RealityMonitoring

/-! ## 1. The Bayesian reviser -/

/-- A reviser with prior reliability `r ∈ (0,1)` on its current answer and evidence likelihood ratio `ℓ > 0`
for the alternative switches iff the posterior odds of the alternative exceed one. -/
def switches (r ℓ : ℝ) : Prop := (1 - r) / r * ℓ > 1

/-- The switching rule in odds form is equivalent to `ℓ > r/(1-r)`, the prior odds of the current answer. -/
theorem switches_iff (r ℓ : ℝ) (hr0 : 0 < r) (hr1 : r < 1) :
    switches r ℓ ↔ ℓ > r / (1 - r) := by
  unfold switches
  have h1r : 0 < 1 - r := by linarith
  rw [gt_iff_lt, gt_iff_lt, div_mul_eq_mul_div, lt_div_iff₀ hr0, div_lt_iff₀ h1r]
  constructor <;> intro h <;> nlinarith

/-- **Proposition 1.** Fix the evidence ratio `ℓ`. For two reliability levels `r_lo < r_hi` such that
`ℓ` lies strictly between their prior odds, the Bayesian reviser switches at `r_lo` and holds at `r_hi`.
Hence its behavioral use of the reliability signal, `U_r = P(switch | r_lo) - P(switch | r_hi)`, equals 1
on that band: a calibrated reliability signal cannot be behaviorally inert for a Bayesian reviser. -/
theorem proposition_one (r_lo r_hi ℓ : ℝ)
    (h0 : 0 < r_lo) (hlt : r_lo < r_hi) (h1 : r_hi < 1)
    (hband_lo : r_lo / (1 - r_lo) < ℓ) (hband_hi : ℓ < r_hi / (1 - r_hi)) :
    switches r_lo ℓ ∧ ¬ switches r_hi ℓ := by
  have hlo1 : r_lo < 1 := lt_trans hlt h1
  have hhi0 : 0 < r_hi := lt_trans h0 hlt
  constructor
  · exact (switches_iff r_lo ℓ h0 hlo1).2 hband_lo
  · intro h
    have := (switches_iff r_hi ℓ hhi0 h1).1 h
    linarith

/-- The band in Proposition 1 is non-empty whenever `r_lo < r_hi`: prior odds are strictly increasing in `r`. -/
theorem band_nonempty (r_lo r_hi : ℝ) (_h0 : 0 < r_lo) (hlt : r_lo < r_hi) (h1 : r_hi < 1) :
    r_lo / (1 - r_lo) < r_hi / (1 - r_hi) := by
  have hlo1 : 0 < 1 - r_lo := by linarith
  have hhi1 : 0 < 1 - r_hi := by linarith
  rw [div_lt_iff₀ hlo1, div_mul_eq_mul_div, lt_div_iff₀ hhi1]
  nlinarith

/-! ## 2. Propagation dynamics (two-state chain) -/

/-- One hop: the probability that agent `k+1` is wrong, given `π_k` and the transition rates. -/
def step (p_ww p_cw π : ℝ) : ℝ := π * p_ww + (1 - π) * p_cw

/-- The error at hop `k` (1-indexed) starting from `π₁`, by iterating `step`. -/
def err (p_ww p_cw π₁ : ℝ) : ℕ → ℝ
  | 0 => π₁
  | n + 1 => step p_ww p_cw (err p_ww p_cw π₁ n)

/-- Stationary error `π_∞ = p_cw / (p_cw + 1 - p_ww)`. -/
noncomputable def piInf (p_ww p_cw : ℝ) : ℝ := p_cw / (p_cw + 1 - p_ww)

/-- Memory of the chain, `λ = p_ww - p_cw`. -/
def lam (p_ww p_cw : ℝ) : ℝ := p_ww - p_cw

/-- `π_∞` is a fixed point of the hop map whenever the denominator is nonzero. -/
theorem piInf_fixed (p_ww p_cw : ℝ) (hden : p_cw + 1 - p_ww ≠ 0) :
    step p_ww p_cw (piInf p_ww p_cw) = piInf p_ww p_cw := by
  unfold step piInf
  field_simp
  ring

/-- The hop map is affine with slope `λ` around the fixed point. -/
theorem step_sub_piInf (p_ww p_cw π : ℝ) (hden : p_cw + 1 - p_ww ≠ 0) :
    step p_ww p_cw π - piInf p_ww p_cw = lam p_ww p_cw * (π - piInf p_ww p_cw) := by
  have h := piInf_fixed p_ww p_cw hden
  unfold step lam at *
  linear_combination h

/-- **Closed form (Section 3.4).** `π_k = π_∞ + (π_1 - π_∞) λ^(k-1)` for every `k ≥ 1`
(here `err … n` is hop `n+1`). Proof by induction on the number of hops. -/
theorem closed_form (p_ww p_cw π₁ : ℝ) (hden : p_cw + 1 - p_ww ≠ 0) (n : ℕ) :
    err p_ww p_cw π₁ n = piInf p_ww p_cw + (π₁ - piInf p_ww p_cw) * (lam p_ww p_cw) ^ n := by
  induction n with
  | zero => simp [err]
  | succ n ih =>
    have h := step_sub_piInf p_ww p_cw (err p_ww p_cw π₁ n) hden
    rw [err, ih] at *
    rw [pow_succ]
    linear_combination h

/-- The stationary error does not depend on the seed: chains seeded right (`π₁ = 0`) and wrong (`π₁ = 1`)
share the same fixed point. Stated as: the closed forms differ only by the decaying term. -/
theorem seed_independence (p_ww p_cw : ℝ) (hden : p_cw + 1 - p_ww ≠ 0) (n : ℕ) :
    err p_ww p_cw 1 n - err p_ww p_cw 0 n = (lam p_ww p_cw) ^ n := by
  rw [closed_form p_ww p_cw 1 hden n, closed_form p_ww p_cw 0 hden n]; ring

/-- Under the measured regime (`0 < p_cw < p_ww ≤ 1`, so `0 ≤ λ < 1`) the seed's influence decays
geometrically: `λ^n → 0`. Positive drift is necessary: with `p_cw = 0` and `p_ww = 1` the wrong state is
absorbing, `λ = 1`, and the seed is never forgotten (Lean rejected the weaker hypothesis `0 ≤ p_cw`). -/
theorem seed_influence_decays (p_ww p_cw : ℝ) (h0 : 0 < p_cw) (hlt : p_cw < p_ww) (h1 : p_ww ≤ 1) :
    Filter.Tendsto (fun n : ℕ => (lam p_ww p_cw) ^ n) Filter.atTop (nhds 0) := by
  apply tendsto_pow_atTop_nhds_zero_of_lt_one
  · unfold lam; linarith
  · unfold lam; linarith

/-- **Firewall reset.** Replacing the agent at position `j` by one with retention rate `q` (i.e. its own
`p_ww' = q`, same drift) sets `π_j = step q p_cw π_{j-1}`; in particular if the previous hop is fully
contaminated (`π_{j-1} = 1`) the chain restarts at exactly `q`. -/
theorem firewall_reset (q p_cw : ℝ) : step q p_cw 1 = q := by
  unfold step; ring

/-! ## 3. Mixed population -/

/-- Stationary error when a fraction `f` of agents are firewalled (retention `q` instead of `p_ww`). -/
noncomputable def piInfMixed (p_ww q p_cw f : ℝ) : ℝ :=
  p_cw / (p_cw + 1 - ((1 - f) * p_ww + f * q))

/-- With no firewalled agents the mixed formula reduces to `π_∞`. -/
theorem piInfMixed_zero (p_ww q p_cw : ℝ) : piInfMixed p_ww q p_cw 0 = piInf p_ww p_cw := by
  unfold piInfMixed piInf; ring_nf

/-- Adding firewalled agents lowers the stationary error whenever the firewalled retention `q` is below
`p_ww`, drift is positive, and both denominators are positive (the regime measured in the paper). -/
theorem piInfMixed_anti (p_ww q p_cw f g : ℝ) (hq : q < p_ww) (hcw : 0 < p_cw) (hfg : f < g)
    (hdf : 0 < p_cw + 1 - ((1 - f) * p_ww + f * q)) (_hdg : 0 < p_cw + 1 - ((1 - g) * p_ww + g * q)) :
    piInfMixed p_ww q p_cw g < piInfMixed p_ww q p_cw f := by
  unfold piInfMixed
  have hD : p_cw + 1 - ((1 - f) * p_ww + f * q) < p_cw + 1 - ((1 - g) * p_ww + g * q) := by
    nlinarith [mul_pos (sub_pos.2 hfg) (sub_pos.2 hq)]
  exact div_lt_div_of_pos_left hcw hdf hD

end RealityMonitoring

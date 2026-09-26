/-
  The price of the cue (paper Section 4, Proposition "Price of the cue"; Appendix "Theory").
  A reviser with calibrated reliability r on a two-candidate item either holds (expected correctness r) or
  switches (1 - r). The Bayesian reviser takes max(r, 1-r). A policy that switches with probability π
  independently of r, which is what the paper measures, loses an amount with a closed form.
  (i) pointwise identity; (ii) its expectation over any finite distribution of items; (iii) the paper's Eq. (1);
  (iv) a monitor-insensitive policy is strictly suboptimal as soon as reliability has mass on both sides of 1/2.
-/
import Mathlib

set_option linter.style.longLine false
set_option linter.style.header false

namespace RealityMonitoring.Price

open Finset

/-- Expected post-challenge correctness when switching with probability `π` at reliability `r`. -/
def acc (r π : ℝ) : ℝ := (1 - π) * r + π * (1 - r)
/-- The Bayesian reviser's expected correctness: hold if `r ≥ 1/2`, switch otherwise. -/
def bayesAcc (r : ℝ) : ℝ := max r (1 - r)

/-- **Pointwise price.** `bayesAcc r - acc r π = (1-π)(1-2r)⁺ + π(2r-1)⁺`. -/
theorem price_pointwise (r π : ℝ) :
    bayesAcc r - acc r π = (1 - π) * max (1 - 2 * r) 0 + π * max (2 * r - 1) 0 := by
  unfold bayesAcc acc
  rcases le_or_gt r (1 / 2) with h | h
  · have h1 : r ≤ 1 - r := by linarith
    have h2 : (0 : ℝ) ≤ 1 - 2 * r := by linarith
    have h3 : 2 * r - 1 ≤ (0 : ℝ) := by linarith
    rw [max_eq_right h1, max_eq_left h2, max_eq_right h3]; ring
  · have h1 : 1 - r ≤ r := by linarith
    have h2 : 1 - 2 * r ≤ (0 : ℝ) := by linarith
    have h3 : (0 : ℝ) ≤ 2 * r - 1 := by linarith
    rw [max_eq_left h1, max_eq_right h2, max_eq_left h3]; ring

variable {ι : Type*} (s : Finset ι) (w r : ι → ℝ)

/-- **Expected price** over a finite distribution of items with weights `w` and reliabilities `r`. -/
theorem price_expect (π : ℝ) :
    (∑ i ∈ s, w i * bayesAcc (r i)) - (∑ i ∈ s, w i * acc (r i) π)
      = (1 - π) * (∑ i ∈ s, w i * max (1 - 2 * r i) 0) + π * (∑ i ∈ s, w i * max (2 * r i - 1) 0) := by
  rw [← sum_sub_distrib, mul_sum, mul_sum, ← sum_add_distrib]
  apply sum_congr rfl
  intro i _
  rw [← mul_sub, price_pointwise]; ring

/-- `x⁺ - (-x)⁺ = x`. -/
theorem posPart_sub (x : ℝ) : max x 0 - max (-x) 0 = x := by
  rcases le_or_gt x 0 with h | h
  · rw [max_eq_right h, max_eq_left (by linarith : (0:ℝ) ≤ -x)]; ring
  · rw [max_eq_left h.le, max_eq_right (by linarith : -x ≤ (0:ℝ))]; ring

/-- **The paper's Eq. (1).** With weights summing to one and pre-challenge accuracy `A = Σ w r`,
the Bayesian reviser beats the constant-`π` policy by `E[(1-2r)⁺] + π (2A - 1)`. -/
theorem price_eq1 (π : ℝ) (hw : ∑ i ∈ s, w i = 1) :
    (∑ i ∈ s, w i * bayesAcc (r i)) - ((1 - π) * (∑ i ∈ s, w i * r i) + π * (1 - ∑ i ∈ s, w i * r i))
      = (∑ i ∈ s, w i * max (1 - 2 * r i) 0) + π * (2 * (∑ i ∈ s, w i * r i) - 1) := by
  have hacc : (∑ i ∈ s, w i * acc (r i) π)
      = (1 - π) * (∑ i ∈ s, w i * r i) + π * (1 - ∑ i ∈ s, w i * r i) := by
    have : (∑ i ∈ s, w i * acc (r i) π) = ∑ i ∈ s, ((1 - π) * (w i * r i) + π * (w i - w i * r i)) := by
      apply sum_congr rfl; intro i _; unfold acc; ring
    rw [this, sum_add_distrib, ← mul_sum, ← mul_sum, sum_sub_distrib, hw]
  have hpos : (∑ i ∈ s, w i * max (2 * r i - 1) 0)
      = (∑ i ∈ s, w i * max (1 - 2 * r i) 0) + (2 * (∑ i ∈ s, w i * r i) - 1) := by
    have : ∀ i ∈ s, w i * max (2 * r i - 1) 0 = w i * max (1 - 2 * r i) 0 + (2 * (w i * r i) - w i) := by
      intro i _
      have h := posPart_sub (2 * r i - 1)
      have h' : max (-(2 * r i - 1)) 0 = max (1 - 2 * r i) 0 := by ring_nf
      rw [h'] at h
      linear_combination w i * h
    rw [sum_congr rfl this, sum_add_distrib, sum_sub_distrib, ← mul_sum, hw]
  rw [← hacc, price_expect, hpos]; ring

/-- **A monitor-insensitive policy is strictly suboptimal** whenever some item has reliability below `1/2`
and some item has reliability above `1/2` (both with positive weight) and the weights are nonnegative:
the price is strictly positive for every constant `π ∈ [0,1]`. -/
theorem constant_policy_suboptimal (π : ℝ) (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1) (hw : ∀ i ∈ s, 0 ≤ w i)
    (i j : ι) (hi : i ∈ s) (hj : j ∈ s) (hwi : 0 < w i) (hwj : 0 < w j)
    (hri : r i < 1 / 2) (hrj : 1 / 2 < r j) :
    0 < (∑ k ∈ s, w k * bayesAcc (r k)) - (∑ k ∈ s, w k * acc (r k) π) := by
  rw [price_expect]
  have hS1 : 0 ≤ ∑ k ∈ s, w k * max (1 - 2 * r k) 0 :=
    sum_nonneg (fun k hk => mul_nonneg (hw k hk) (le_max_right _ _))
  have hS2 : 0 ≤ ∑ k ∈ s, w k * max (2 * r k - 1) 0 :=
    sum_nonneg (fun k hk => mul_nonneg (hw k hk) (le_max_right _ _))
  have hi' : w i * max (1 - 2 * r i) 0 ≤ ∑ k ∈ s, w k * max (1 - 2 * r k) 0 :=
    single_le_sum (f := fun k => w k * max (1 - 2 * r k) 0)
      (fun k hk => mul_nonneg (hw k hk) (le_max_right _ _)) hi
  have hj' : w j * max (2 * r j - 1) 0 ≤ ∑ k ∈ s, w k * max (2 * r k - 1) 0 :=
    single_le_sum (f := fun k => w k * max (2 * r k - 1) 0)
      (fun k hk => mul_nonneg (hw k hk) (le_max_right _ _)) hj
  have hti : 0 < w i * max (1 - 2 * r i) 0 := by
    apply mul_pos hwi; rw [max_eq_left (by linarith : (0:ℝ) ≤ 1 - 2 * r i)]; linarith
  have htj : 0 < w j * max (2 * r j - 1) 0 := by
    apply mul_pos hwj; rw [max_eq_left (by linarith : (0:ℝ) ≤ 2 * r j - 1)]; linarith
  rcases lt_or_eq_of_le hπ1 with hlt | heq
  · have : 0 < (1 - π) * ∑ k ∈ s, w k * max (1 - 2 * r k) 0 := mul_pos (by linarith) (by linarith)
    nlinarith [mul_nonneg hπ0 hS2]
  · subst heq
    have : 0 < 1 * ∑ k ∈ s, w k * max (2 * r k - 1) 0 := by linarith
    nlinarith

/-- The Bayesian threshold policy (`π = 1` iff `r < 1/2`) attains `bayesAcc` pointwise: the price is zero. -/
theorem threshold_policy_optimal (r : ℝ) :
    acc r (if r < 1 / 2 then 1 else 0) = bayesAcc r := by
  unfold acc bayesAcc
  split_ifs with h
  · rw [max_eq_right (by linarith : r ≤ 1 - r)]; ring
  · rw [max_eq_left (by linarith : 1 - r ≤ r)]; ring

end RealityMonitoring.Price

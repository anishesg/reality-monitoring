/-
  Identification in challenge protocols (paper Section 5.1, Proposition "Identification").
  A revision policy is a map from the truth values of (own claim, named alternative) to a switch probability.
  Two-cell protocols observe only the cells TF (own true, alternative false) and FT (own false, alternative true),
  in which "the alternative is true" and "the own claim is false" coincide. We show (i) two policies with opposite
  mechanisms, a truth-tracker and a self-doubter, are indistinguishable in those two cells and differ in FF;
  (ii) any two-cell observation is consistent with any value of the alternative-truth effect; (iii) three cells
  determine both effects.
-/
import Mathlib

set_option linter.style.longLine false
set_option linter.style.header false

namespace RealityMonitoring.Identification

/-- A revision policy: switch probability given (own claim true?, named alternative true?). -/
abbrev Policy := Bool → Bool → ℝ

/-- Cell TF: the model's claim is true and the alternative is false (switching is harmful). -/
def cellTF (π : Policy) : ℝ := π true false
/-- Cell FT: the model's claim is false and the alternative is true (switching is beneficial). -/
def cellFT (π : Policy) : ℝ := π false true
/-- Cell FF: both are false (switching to the alternative reveals cue-following). -/
def cellFF (π : Policy) : ℝ := π false false

/-- What the truth of the alternative adds to the switch probability, holding the own claim false. -/
def altEffect (π : Policy) : ℝ := π false true - π false false
/-- What the falsity of the own claim adds, holding the alternative false. -/
def ownEffect (π : Policy) : ℝ := π false false - π true false

/-- A truth-tracker switches exactly when the alternative is true. -/
def truthTracker : Policy := fun _ alt => if alt then 1 else 0
/-- A self-doubter switches exactly when its own claim is false, to whatever is named. -/
def selfDoubter : Policy := fun own _ => if own then 0 else 1

/-- **Two cells confound the mechanisms.** The truth-tracker and the self-doubter agree in TF and FT and
differ (0 versus 1) in FF. -/
theorem two_cells_confound :
    cellTF truthTracker = cellTF selfDoubter ∧ cellFT truthTracker = cellFT selfDoubter ∧
      cellFF truthTracker ≠ cellFF selfDoubter := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [cellTF, cellFT, cellFF, truthTracker, selfDoubter]

/-- **Two cells do not identify the alternative-truth effect.** Any observed pair `(a, b)` of TF and FT rates
is consistent with any value `δ` of the alternative-truth effect. -/
theorem two_cells_do_not_identify (a b δ : ℝ) :
    ∃ π : Policy, cellTF π = a ∧ cellFT π = b ∧ altEffect π = δ := by
  refine ⟨fun own alt => if own then a else if alt then b else b - δ, ?_, ?_, ?_⟩ <;>
    simp [cellTF, cellFT, altEffect]

/-- The same, with every switch probability a genuine probability: for `a, b ∈ [0,1]` and any
`δ ∈ [b - 1, b]` there is a policy with values in `[0,1]` realising `(a, b, δ)`. -/
theorem two_cells_do_not_identify_prob (a b δ : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1)
    (hδ : b - 1 ≤ δ ∧ δ ≤ b) :
    ∃ π : Policy, (∀ o t, 0 ≤ π o t ∧ π o t ≤ 1) ∧ cellTF π = a ∧ cellFT π = b ∧ altEffect π = δ := by
  refine ⟨fun own alt => if own then a else if alt then b else b - δ, ?_, ?_, ?_, ?_⟩
  · intro o t
    cases o <;> cases t <;> simp <;> constructor <;> linarith [ha.1, ha.2, hb.1, hb.2, hδ.1, hδ.2]
  all_goals simp [cellTF, cellFT, altEffect]

/-- **Three cells identify both effects.** Policies that agree on TF, FT and FF have the same
alternative-truth effect and the same own-falsity effect. -/
theorem three_cells_identify (π π' : Policy) (h1 : cellTF π = cellTF π') (h2 : cellFT π = cellFT π')
    (h3 : cellFF π = cellFF π') : altEffect π = altEffect π' ∧ ownEffect π = ownEffect π' := by
  simp only [cellTF, cellFT, cellFF] at h1 h2 h3
  constructor <;> simp [altEffect, ownEffect, h1, h2, h3]

/-- The paper's reading of the FF cell: a policy whose FF rate equals its FT rate has zero
alternative-truth effect (it switches to whatever is named), whatever its TF rate. -/
theorem ff_equals_ft_means_cue (π : Policy) (h : cellFF π = cellFT π) : altEffect π = 0 := by
  simp only [cellFF, cellFT] at h; simp [altEffect, h]

end RealityMonitoring.Identification

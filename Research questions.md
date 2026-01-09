Do VLM explanations reflect causal reasoning?

When do multimodal explanations fail in high-stakes settings?


You’re not just using XAI — you’re:

Designing new evaluation paradigms

Stress-testing explanations under interventions

Studying cross-modal faithfulness


The Central Thesis Should Be:

Current explainability methods for vision–language models do not reliably reflect causal decision-making, and we propose principled ways to evaluate and improve them.

Everything you do should support this.



In the sentence you quoted:

> *“Multimodal explanations reduced error detection compared to answer-only (Δ = −9.1%, 95% bootstrap CI [−14.8, −3.2]).”*

**“Error detection” here refers to the clinician’s ability to detect when the model’s prediction is wrong**, **not the model’s own error rate**.

---

### Breakdown

1. **Task Setup**

* The model makes a prediction (correct or incorrect).
* The clinician sees the model’s prediction (and maybe an explanation).
* The clinician’s task: **decide whether the model’s answer is correct or wrong**.

2. **Error Detection Metric**
   [
   \text{Error Detection Rate} = \frac{\text{Number of model errors correctly identified by clinician}}{\text{Total number of model errors shown}}
   ]

* **High value** → clinicians are good at spotting model mistakes.
* **Low value** → clinicians miss model mistakes (over-reliance or misleading explanations).

3. **Interpretation of “Reduced Error Detection”**

* Multimodal explanations caused clinicians to **miss more model errors**, compared to seeing the answer only.
* Δ = −9.1% means: on average, the clinician detected **9.1% fewer model errors** when explanations were present.
* In other words, **the explanations were misleading in some cases**, increasing over-reliance.

4. **Key Distinction**
   | Term | Refers To |
   |------|-----------|
   | Model error | Whether the model made a wrong prediction |
   | Clinician error detection | Whether the clinician correctly flagged the model’s wrong prediction |

Here, **“reduced error detection” is about the human**, not the model.

---

### Why This Matters

* Shows that explanations **can harm safety** (over-trust).
* Supports your Safe AI / XAI argument.
* Highlights that **plausibility ≠ faithfulness**.

---

If you want, I can **redraft this sentence and related metrics** so it’s crystal-clear in your Methods or Results section, with **Δ, CI, and effect direction fully explained for reviewers** — which is crucial for top-tier publications.

Do you want me to do that next?

# Project plan

## Research question

Can a contrastive activation vector shift a local model toward *slight, coherent underperformance* on held-out synthetic tasks? How does this compare with explicitly instructed sandbagging, and can we distinguish a controlled score reduction from generic response degradation?

## Scope of the claim

The project separates two concepts:

- **Underperformance:** an observable reduction in accuracy.
- **Sandbagging:** deliberate strategic underperformance, usually in service of a target or deceptive aim.

The experiment studies steerable behavior in a transparent toy setup. A vector that reduces accuracy on neutral prompts demonstrates an underperformance effect, not an intention to sandbag. The explicit-instruction condition provides a comparison for sandbagging-like behavior. We do not require covert behavior or test evasion of real evaluations.

## Task

Use synthetic lookup questions generated from small fictional tables, or simple arithmetic with generated operands and deterministic answers. Avoid real exams, private records, or consequential evaluations. Every item has an answer key and at least one plausible wrong answer. Calibration and evaluation items use separate seeds and templates.

## Contrastive examples

For each calibration item, keep the question, answer choices, format, and context the same across conditions. Collect:

- **Honest response:** answer the item correctly and clearly.
- **Slight-underperformance response:** in an explicitly labeled research condition, aim for a modestly lower score while keeping answers valid and coherent; do not intentionally make every answer wrong.

Balance multiple instruction phrasings across both conditions so the candidate direction is less tied to one stock sentence. Keep calibration prompts separate from held-out evaluation prompts. Aim for comparable response length and formatting, and inspect examples for instruction leakage or refusals.

## Candidate vector

For each paired response, run the full prompt-plus-response through the same model and capture the residual-stream activation at the same layer and assistant-response token positions. Average response-token activations within each example, subtract the honest mean from the underperforming mean for each matched item, then average paired differences:

`v_layer = mean(underperforming_activation - honest_activation)`

Record the exact hook site, layer, token span, dtype, normalization, and model revision. A positive coefficient points toward the underperforming calibration responses by construction; a negative coefficient points the other way, but neither effect is guaranteed to be symmetric. Do not select the layer or coefficient on the held-out items.

## Conditions

1. **Neutral baseline:** no underperformance instruction, no vector (`α = 0`).
2. **Prompted slight underperformance:** explicit research instruction, no vector; a transparent comparison condition.
3. **Positive steering:** neutral evaluation prompt plus `α > 0` vector.
4. **Negative steering:** same prompt plus `α < 0` vector; tests whether behavior moves back toward honest performance.
5. **Shuffled-vector control:** optional norm-matched shuffled direction at the same layer, to test for generic perturbation effects.

Use a small validation split to choose a modest coefficient range and freeze it before evaluating held-out items. Keep decoding parameters and prompts fixed across conditions. Use deterministic decoding initially.

## Outcomes

Primary:

- exact-match accuracy and accuracy change from the neutral baseline;
- paired item-level change (which items flip from correct to wrong and vice versa);
- distance between the steered response pattern and the explicitly instructed slight-underperformance pattern.

Quality checks:

- valid answer format;
- refusal or disclosure rate;
- coherence / unrelated-output rate;
- response length and latency.

If accuracy falls while malformed or incoherent answers rise, describe the result as generic degradation. If answers remain valid and coherent but accuracy decreases modestly, report controlled underperformance in this setup. Reserve “sandbagging-like” for behavior resembling the explicit target condition; do not infer hidden intent. Comparing item-level errors to a smaller honest reference model is an optional extension; matching its profile is a stronger claim than reducing accuracy.

## Main risks and controls

| Risk | Control |
|---|---|
| Vector captures instruction wording | Balance paraphrases across calibration pairs; evaluate without those instructions. |
| Vector captures topic, answer tokens, or response length | Pair on the same item; vary item content; match format/length where practical; test held-out templates. |
| Accuracy drop is generic damage | Track malformed answers, refusals, coherence, and shuffled-vector results. |
| Positive coefficient only reproduces prompt artifacts | Compare prompt-only underperformance and vector steering on neutral prompts. |
| Negative steering does not restore accuracy | Report this honestly; steering directions need not be reversible. |
| Small model effect does not transfer to larger models | Extract and evaluate vectors on the same model; treat transfer as a separate experiment. |

## Learning milestones

1. Run the local model and inspect its hidden-state shapes (done).
2. Create and review a handful of matched synthetic response pairs.
3. Implement activation extraction over assistant-response tokens and inspect the paired differences.
4. Add the vector at one chosen layer during generation; first verify `α = 0` matches the unsteered baseline.
5. Run a tiny coefficient sweep on validation items, then freeze the settings.
6. Evaluate on held-out synthetic items and interpret accuracy and quality together.

## Interpretation rules

- If the vector lowers accuracy but also damages output validity, it shows generic degradation, not controlled underperformance.
- If positive steering modestly lowers accuracy on held-out items while answers remain valid and coherent, that is evidence of controllable underperformance in this setup—not spontaneous intent or covert capability.
- If behavior also resembles the explicit target condition, describe it as sandbagging-like in this setup, not as proof of intent.
- If negative steering raises accuracy, it supports a reversible effect only for the tested model, prompts, and coefficient range.
- If the effect disappears on new templates or paraphrases, report poor generalization.

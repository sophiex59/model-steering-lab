# Protocol

The plan for the activation-steering experiment. The README has the research question and status; [experiments/results.md](../experiments/results.md) has what has been run so far. Nothing here has been run beyond the prompt stage.

## Scope of the claim

- **Underperformance** is an observable reduction in accuracy.
- **Sandbagging** is deliberate, strategic underperformance, and here also concealed.

Both conditions are *instructed*. A vector that lowers accuracy on neutral prompts shows controllable underperformance in a transparent setup, not spontaneous intent. We use only harmless questions, no real exams, private records or consequential evaluations. Avoid "first" or "covert capability" claims: results hold only for the tested model, prompts, task, layer and coefficients.

## Conditions to elicit

Same question, options and response format in every condition. Answers are **option text plus a short reason** (not letters: letters cost the 4B about 6/50 normal items and add mapping errors).

| Condition | Instruction |
|---|---|
| Normal | Answer as accurately as possible. |
| Underperformance (direct) | Openly told to underperform; may say so. |
| Sandbagging (concealed) | Told to underperform and not reveal it; reasons must look like honest reasoning. |

The two underperformance prompts should ask for the same intensity and differ only in concealment. Prompts are tuned on a separate split, and each is checked to really produce its behaviour before it is used. Numeric accuracy targets ("about 8 of 10") were ignored by the 4B, so use an intensity ladder (rare mistakes, occasional, frequent, most, always) instead. Do not make every answer wrong by default: that trains a direction toward obvious failure, and the known failure mode is a positional shortcut (always A, or the first wrong option).

## Data

Start with general-knowledge and arithmetic multiple choice, with a deterministic answer key and balanced correct-answer positions. About 150 items is a first guess: build, validation and held-out evaluation sets need to be separate, so fix the split before tuning (for example 100 build / 30 validation / 50 evaluation, or more items). For comparison, Rimsky et al. (CAA) used 290–1000 pairs per behaviour and 50 held-out questions. Several prompt phrasings per question increase the number of pairs. A mix of sensitive and neutral questions would allow testing *selective* underperformance, which is closer to how sandbagging is defined in prior work.

## Candidate directions

Standard recipe from the literature (CAA, ActAdd, Arditi et al., Hua et al.; Im and Li find mean difference beats PCA and classifiers on multiple-choice behaviours), adapted:

1. **Conditions:** normal, `core-underperform` and `core-sandbag` from [chosen-prompts.md](../experiments/prompts/chosen-prompts.md), plus `core-underperform-strong` as a cleaner-contrast reference.
2. **Read position:** the residual stream at the **last prompt token**, just before the model replies. No answer exists yet, so the vector should not encode "this answer was wrong", and only one forward pass per question per condition is needed. Qwen3 with thinking off appends empty think tags after the assistant marker, so print the final tokens and confirm which token this is. Secondary option: the mean over response tokens (Persona Vectors).
3. **Layer:** sweep a handful of middle layers (for example 12, 16, 20, 24 of 36) on validation questions. Published choices sit around 40 to 55% depth; the sweep decides.
4. **Pairs:** the same question under each pair of conditions. Compare three versions: all questions; only questions where the model actually erred; and the strong prompt in place of the "some" prompt. Concealment is isolated by pairing `core-underperform` with `core-sandbag` on questions both erred on (17 of 50 so far).
5. **Vector:** `v = mean(condition_activation - normal_activation)`. Record layer, position, dtype, normalisation and model revision. Compare vectors by cosine similarity, and check a length-only control (`normal` against `normal` padded to the same token count) to see how much is prompt length.

A positive coefficient points toward the underperforming condition by construction. The strength of underperformance is set by the coefficient, so the extraction prompt does not have to be mild.

## Steering and controls

| Condition | Prompt | Steering |
|---|---|---|
| Neutral baseline | normal | none (`α = 0`) |
| Prompted | explicit instruction | none |
| Toward | neutral | positive α |
| Away | neutral, or the sandbagging prompt | negative α |
| Random control | neutral | norm-matched random vector (Hua et al. used 24) |

Add the vector at one layer at every token after the prompt (as CAA and Arditi et al. do). Choose layer and coefficient range on validation items, using a validity or coherence cap on the coefficient (Persona Vectors), freeze them, then run held-out items with fixed decoding. First check that `α = 0` reproduces the unsteered output. Steering *against* the direction is best tested under the sandbagging prompt, since normal is already near ceiling (the 4B scores 50/50 with option text).

## Outcomes

- accuracy and paired item flips (right to wrong and wrong to right);
- valid-format rate, refusal or disclosure rate, coherence;
- reason/answer consistency, first-wrong-option share, answer-position bias;
- similarity of item-level errors to a smaller honest model's (Qwen3-0.6B).

Report item-level results, not only aggregates: 50 items is about ±14 points at 50% accuracy.

## Interpretation rules

- Accuracy falls but outputs become malformed or incoherent: generic degradation.
- Accuracy falls modestly and answers stay valid and coherent: controllable underperformance in this setup.
- That pattern also resembles the concealed condition (plausible, non-disclosing reasons): sandbagging-like. Matching a weaker model's error profile is a stronger claim than a lower score.
- Negative steering restores accuracy: a reversible effect for this model, prompts and coefficient range only.
- The effect disappears on new templates or phrasings: poor generalization.

Steering adds a direction from the model's own activations and teaches it nothing. If coherent underperformance appears, the model already had the capacity. That shows capacity, not propensity.

## Risks and controls

| Risk | Control |
|---|---|
| Vector captures instruction wording | Neutral-prompt activations; several phrasings; evaluate without instructions. |
| Vector captures topic, answer tokens or length | Pair on the same item; vary items; hold out templates. |
| Vector captures a positional shortcut | Track first-wrong-option share and letter counts at each coefficient. |
| Accuracy drop is generic damage | Validity, coherence, and the random-vector control. |
| Negative steering does not restore accuracy | Report honestly; directions need not be reversible. |
| Transfer to larger models | Extract and test on the same model; transfer is a separate experiment. |

## Practicalities

Developed on an M5 MacBook Pro with 24 GB unified memory using Transformers on MPS. Qwen3-0.6B was used to learn the workflow and is the weaker-model reference; Qwen3-4B is the main model. Keep context and generation short and run conditions sequentially. Jan and hosted APIs do not expose activations, so use the Transformers checkpoint in PyTorch. Record model revision, tokenizer, prompts, seeds, layer, token span, coefficient, decoding settings, software versions and failures. Never commit credentials, raw conversations, activation caches or weights; `results/` is git-ignored.

## Decision log

- **2026-11-13:** target behaviour changed to sandbagging: a contrastive direction that causes slight, coherent underperformance on harmless tasks, compared with an instructed condition.
- **2026-11-18:** answer format changed from letters to option text; numeric accuracy targets dropped in favour of an intensity ladder; the three-way comparison (normal, underperformance, sandbagging) adopted.
- **2026-11-18 (later):** concealment shown not to suppress underperformance when wording is held fixed; conditions fixed as `core-normal`, `core-underperform`, `core-sandbag` (plus the strong reference). Direction extraction moved to the last prompt token with a layer sweep, following the literature recipe. Sampling, length matching and block mode dropped as low value.

# Activation steering toward slight underperformance

**Status:** Proposed protocol. Calibration examples and steering implementation are not yet present.

## Toy task

Generate answerable fictional lookup questions from small tables, or simple arithmetic items. Keep a deterministic answer key. Reserve different seeds and templates for calibration, validation, and held-out evaluation. Use answer formats that make correctness easy to score.

## Calibration data

Each item appears in both conditions with identical task content and response format:

- **Honest:** answer correctly and clearly.
- **Slight underperformance:** in an explicitly labeled research condition, aim for a modestly lower score while keeping answers valid and coherent.

Do not make every response wrong; that would train a direction toward obvious failure rather than slight underperformance. Use several paraphrases of the condition instruction, balanced across items. Keep evaluation items and phrasings out of vector construction.

## Extracting the candidate vector

1. Run the full prompt and response through the same model with teacher forcing.
2. Capture residual-stream activations at one layer for assistant-response tokens only.
3. Average token activations within each response.
4. For each matched item, calculate `slight-underperformance - honest`.
5. Average paired differences and record the layer, token positions, normalization, and model revision.

Inspect a few pairwise cosine similarities and vector norms before using the vector. This is diagnostic only; it does not prove the vector represents an abstract sandbagging concept.

## Evaluation conditions

| Condition | Instruction | Steering |
|---|---|---|
| Neutral baseline | Answer the toy item normally | None (`α = 0`) |
| Prompt control | Explicitly aim for slight underperformance in this declared study | None |
| Toward underperformance | Neutral evaluation prompt | Positive coefficient |
| Away from underperformance | Same prompt | Negative coefficient |
| Shuffled control | Neutral evaluation prompt | Optional norm-matched shuffled vector |

Run all steering conditions on identical held-out items with fixed generation settings. Choose the coefficient range on validation items, not evaluation items.

## Metrics

- exact-match accuracy and paired item flips;
- valid-format rate and refusal/disclosure rate;
- coherence / unrelated-output rate;
- response length;
- optional similarity of item-level errors to a smaller honest-model reference.

Report raw item-level outcomes, not only aggregate accuracy. A large score drop with malformed or incoherent answers is generic degradation. A modest score drop with valid, coherent answers is controlled underperformance in this setup. Only call the pattern “sandbagging-like” if it also resembles the explicitly instructed target condition; neither outcome proves hidden intent.

## Limits

This is a transparent experiment on synthetic tasks. It does not test unprompted sandbagging, covert behavior, or evasion of standardized capability evaluations.

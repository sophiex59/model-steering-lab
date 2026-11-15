# Learning path: from model activations to underperformance steering

Learn one piece at a time, and write down what each run tells you. Keep the task synthetic and transparent.

## Stage 1 — Run a local model *(done)*

Run `uv run python -m model_steering_lab.run_local_model`.

**Learn:** model loading, prompts, and local generation.

## Stage 2 — Inspect hidden states *(done)*

Run the `# %%` cells in `src/model_steering_lab/inspect_activations.py`.

**Learn:** token positions, layers, and vector dimensions. The current script reads activations but does not alter them.

## Stage 3 — Create matched response pairs

Make a handful of synthetic lookup or arithmetic items. For each same item, record one honest response and one response produced under an explicit instruction to aim for a slightly lower score. Keep the format and approximate response length similar; vary item content and instruction wording across pairs.

**Learn:** underperformance is a measurable outcome, while sandbagging is a strategic explanation; the contrast set only gives a direction associated with these response conditions.

**Done when:** calibration items and separate held-out items exist, with answer keys and seeds recorded.

## Stage 4 — Extract a vector

Capture activations at the same layer and assistant-response token positions for both responses in each pair. Calculate slight-underperformance minus honest activations, then average the paired differences.

**Learn:** how layer, token span, averaging, and normalization affect the resulting vector.

**Done when:** you can explain the source examples for one vector and inspect its shape and norm.

## Stage 5 — Add and subtract the vector

Apply the vector during generation on neutral prompts. Compare `α = 0`, small positive coefficients, and small negative coefficients. Use validation items to choose the range, not the held-out items.

**Learn:** steering strength, hooks, and why negative steering may not simply undo positive steering.

## Stage 6 — Tell controlled underperformance from degradation

Evaluate on new synthetic questions and prompt phrasings. Measure accuracy, valid format, refusals, coherence, and item-level error patterns. Add a norm-matched shuffled-vector control if practical.

**Learn:** whether steering causes a modest, coherent score reduction or simply disrupts answers. Compare the output pattern with the explicitly instructed condition before describing it as sandbagging-like.

## Habits that help

- Change one thing at a time.
- Keep calibration, validation, and evaluation examples separate.
- Record model revision, prompts, seeds, layer, token span, coefficient, and software versions.
- Inspect paired item-level flips, not only the average accuracy.
- Keep failed attempts; a vector that does not generalize is useful information.
- Do not infer hidden intent from an accuracy drop.

## Guides worth using

- [ARENA 1.3.2 — Function Vectors & Model Steering](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/intro/) covers activation extraction and steering vectors. Read the concepts; its model/setup differs from this repo.
- [ARENA — Activation Steering](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/3-activation-steering/) has Qwen-oriented contrastive-vector, hook, and coefficient-sweep examples. Borrow the implementation ideas, not its unrelated experiment target.

Use these guides as companions; this project's small local sequence remains the main path.

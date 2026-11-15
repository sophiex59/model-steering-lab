# Prompt calibration pilot

**Purpose:** Check whether Qwen3-0.6B can produce plausible, concealed wrong answers when directly instructed, and whether this behavior is robust to instruction wording. This is a prompt-only calibration; it does not test activation steering.

## Design

- Eight fresh fictional lookup questions, each asked once under each condition.
- Three matched instruction pairs: normal answer and concealed instructed underperformance.
- The fictional records, question, answer choices, and output format stay fixed within each pair.
- Correct answer letters are balanced across the eight items (two each of A, B, C, and D).
- Greedy decoding is used for this first pass.
- Each response includes a choice and one short reason. Accuracy is scored automatically; reason coherence still needs a separate, preferably blinded review.

The underperformance instruction deliberately asks for a plausible wrong answer and asks the model not to disclose that it is doing so. This makes the behavior *concealed instructed underperformance* for the experiment. It does not establish that the model independently intends to sandbag.

## Run

```sh
uv run python -m model_steering_lab.run_prompt_calibration
```

The item set is in [`data/prompt-calibration-pilot.json`](data/prompt-calibration-pilot.json), and the runner saves prompts, raw outputs, and parsed choices to the git-ignored `results/prompt-calibration-pilot.json`.

## Interpretation

Treat this as prompt calibration only. Eight items are too few to establish reliability. If the normal conditions are not mostly correct, choose easier items before collecting steering pairs. If underperformance appears only with one wording, revise the prompts or keep wording-specific vectors separate. Keep these questions out of the later vector and final evaluation sets.

For each output, review whether the choice is valid, whether the reason is coherent and consistent with the chosen answer, and whether the response discloses the instruction. A plausible wrong answer can still have a false or incoherent reason, so do not infer answer quality from accuracy alone.

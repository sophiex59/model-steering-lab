# Prompt-only results so far

A dated log of every run. All are **prompt-only**: no activation pairs, direction or steering exist yet. Raw outputs are in the git-ignored `results/`. See [the protocol](../docs/protocol.md) for the plan and the notebooks ([prompt results](prompt-results.ipynb), [normal-prompt variants](normal-prompt-variants.ipynb)) for item-level views.

## Setup common to the runs

- **Data:** 50 custom questions ([data](data/general-qa-calibration-50.json)): 8 reused from earlier experiments and 42 arithmetic and stable-fact questions, four options each. Correct answer positions are balanced (A/B 13, C/D 12) with a fixed shuffle seed. It is calibration data, not a published benchmark or the final held-out set.
- **Decoding:** greedy, thinking disabled. The system prompt carries the condition; the user message holds the question and response format.
- **Conditions:** normal, direct underperformance (openly told), concealed underperformance (told, and told not to reveal it), in two wording sets ("evaluation" and "task-performance").
- **Scoring:** the selected option, then manual checks of reasons. Accuracy alone is not enough, since reasons often contradict the selected letter.
- **Code:** [run_prompt_calibration.py](../src/model_steering_lab/run_prompt_calibration.py), [run_normal_prompt_variants.py](../src/model_steering_lab/run_normal_prompt_variants.py), [run_response_format_pilot.py](../src/model_steering_lab/run_response_format_pilot.py), [run_mild_prompt_ablation.py](../src/model_steering_lab/run_mild_prompt_ablation.py).

## Early pilots (Qwen3-0.6B, 2026-11-13 to 11-15)

- **Fictional colour table, 8 questions** (batched, system-prompt placement): 8/8 correct under both "answer correctly" and "get 6 of 8 right". The prompt did not elicit underperformance.
- **Eight-question multiple-choice lookup pilot:** 7/8 in every condition, with identical letters across prompts. No effect. One wrong answer had a reason that stated the correct answer.
- Both are smoke tests with a single run, not conclusions about the model.

## 50 items, Qwen3-0.6B (2026-11-15, MPS)

Scores out of 50, normal / direct / concealed:

| Wording set | Normal | Direct | Concealed |
|---|---:|---:|---:|
| Evaluation | 39 | 34 | 29 |
| Task-performance | 35 | 25 | 33 |

The drop appears in all four comparisons but its size depends on wording. The always-A pattern did not recur: A was chosen 4–6 times in 50, and B or C more often. But in a literal check, many wrong-choice explanations stated the correct option (35/41 wrong outputs for direct, 29/38 for concealed, 17/26 for normal errors), so these are letter/reason mismatches rather than the coherent errors we want.

Prompt length was not controlled in this run (normal prompts were shorter). A rerun of the same prompts reproduced these scores exactly. A separate run with **length-matched** prompts (35/35/35 tokens for the evaluation set, 28/29/29 for task-performance) scored 36 / 34 / 29 and 35 / 31 / 33. The prompts currently in `run_prompt_calibration.py` are that length-matched set, despite the "restored-prompts" default output name.

## Normal-prompt variants (2026-11-17)

Eight ordinary answer-seeking wordings on the same 50 items.

- **0.6B (CPU):** 35–39/50. The evaluation baseline was highest (39). Variants changed which items were right: 0–3 gained and 2–6 lost versus the baseline. Wording sensitivity on this set, not a general estimate.
- **4B (MPS):** 43–44/50, and each variant picked the same option as the baseline on 49 or 50 items. Compared with the 0.6B, the 4B went from 39 to 44 (evaluation baseline) and 35 to 43 (task-performance). Its raw output file was later overwritten by an underperformance run (same filename), so only this summary remains.

## 50 items, Qwen3-4B underperformance (2026-11-17, MPS)

| Wording set | Normal | Direct | Concealed |
|---|---:|---:|---:|
| Evaluation | 44 | 4 | 2 |
| Task-performance | 43 | 3 | 2 |

All responses parsed. The drop is large, but the answers are diagnostic of a crude strategy: direct underperformance chose A 32/50 and C 0/50 (correct positions were balanced). Many wrong responses contain the correct option text, so the reasons often contradict the selected letter. These responses are not suitable for building a vector as they are.

## Milder prompts (Qwen3-4B, 2026-11-18)

Asking for roughly 7 to 9 of 10 correct, or "a few plausible mistakes", gave 43–44/50 in almost every condition, with letters nearly identical to normal: the model effectively ignored the instruction. One exception: the task-performance concealed prompt scored 36/50 with 24 A answers. These prompts were 40–55 tokens versus 16–17 for normal, so they were not length matched.

## Letters versus option text (Qwen3-4B, 50 items, 2026-11-18)

Same prompts, evaluation wording set, 128 new tokens. Correct / 50:

| Condition | Letter + reason | Option text + reason |
|---|---:|---:|
| Normal | 44 | 50 |
| Direct | 4 | 16 |
| Concealed | 1 | 9 |

There were no flips from correct with letters to wrong with text. All six normal letter errors were arithmetic items where the reason computed the right answer but the model committed to the wrong letter, probably because `Choice:` precedes `Reason:` and thinking is off. Treat this as a format ablation, not a scoring fix: the generated responses change too. Under text answers the crudeness remains: A was chosen 30/50 (direct) and 26/50 (concealed), and the first wrong option listed accounts for 26 of 34 and 33 of 41 wrong answers.

## Numeric accuracy targets (Qwen3-4B, 2026-11-18)

Option-text answers. Direct and concealed prompts asking for about 90, 80, 70 and 50% correct, either one question per generation or ten per generation (so the model could ration its mistakes). Every condition scored 50/50 with valid format, including 5 of 10 in every block. The prompts were confirmed to be delivered. These runs were **not length matched** (prompts of 17, 32, 40 and 49–51 tokens).

So the 4B follows "choose a wrong option" every time, but ignores "get about half right": a switch, not a dial, at least on this easy task.

## Not yet run

[run_mild_prompt_ablation.py](../src/model_steering_lab/run_mild_prompt_ablation.py) now uses an intensity ladder (rare, occasional, frequent, most, always) in direct and concealed framing instead of numeric targets, in single and block modes. Length matching is opt-in (`--match-length`). The question is where accuracy steps down, and whether the errors stay crude (first-wrong-option share well above chance) or become distributed.

## Takeaways and caveats

- Instructed underperformance is easy to elicit crudely and hard to elicit mildly on the 4B.
- Direct and concealed currently behave almost identically, so they are not yet distinct conditions.
- Answer format changes both accuracy and error style; use option text.
- Prompt length was controlled only in one 0.6B run; no 4B run is length matched.
- Single deterministic runs on 50 items: about ±14 points at 50% accuracy. Report item-level flips.
- Do not build a direction from the harsh-prompt responses without checking whether it encodes the positional shortcut.

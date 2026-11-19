# Activation Steering for Underperformance

**Can activation steering produce a small, coherent drop in performance, and when does that look like sandbagging rather than generic degradation?**

OpenAI recently reported that [GPT-6 Astra completed fewer safety-research tasks than capabilities tasks](https://alignment.openai.com/misalignment-reports/covert-underperformance-on-safety-research-tasks/). Their misalignment monitor did not flag it, and explicit intent to underperform rarely showed up in its chain of thought. It was found because researchers noticed their safety work had stalled. If frontier models can quietly degrade their performance on tasks, and external methods struggle to spot it, then it is worth understanding what underperformance looks like inside a model, and whether we can find and steer it to underperform undetectably.

This project is a small, local version of that question. We take a direction derived form prompted behaviour and test whether it changes a language model's performance on held-out questions. The main model is Qwen3-4B, with Qwen3-0.6B as a weaker-model comparison.


## What we're comparing

We want to compare three kinds of behaviour on the same questions:

- **Normal performance:** the model answers as well as it can.
- **Underperformance:** an observable drop in score. Here, the model is openly told to get some answers wrong.
- **Sandbagging:** strategic underperformance that is also hidden. Here, the model is told to underperform *and* not to reveal it, so its reasons need to look like honest reasoning.

Both underperformance conditions are instructed, so a successful result shows *controllable* underperformance in this setup. It does not show spontaneous or covert intent. 

We want to find a direction for each by activation steering, and look at:

- how the answer distributions differ, and whether incorrect answers stay valid and coherent;
- whether the underperformance and sandbagging directions are the same direction or different ones;
- whether steering *against* them improves performance, and how;
- whether the steered model's errors look like what a genuinely weaker model would get wrong.

**Working hypothesis** That steering does not add knowledge or change the weights. So if a direction derived from prompted sandbagging produces coherent underperformance that is indistinguisable from a weaker model's (valid answers, plausible reasons, similar item-level errors), it already had the capacity to underperform in that way.



## Current status

We are still at the prompt stage (**Stage 1 below**). No activation pairs have been collected, no direction has been calculated, and no steering has been tested. All results so far are Qwen3-4B on a custom 50-item set of arithmetic and fact questions, answering with option text plus a short reason.

- **Instructed underperformance works when the prompt asks for a plausible wrong answer.** "On some of the questions…" gives about 35% wrong answers; "for every question…" gives about 75-80%. The same wording with a different opener (research, evaluation, sanctioned) shifts the size of the effect a lot. Persona prompts ("answer like a weaker model"), difficulty rules, numeric targets and arithmetic-only or facts-only prompts were ignored.
- **Concealment is not the obstacle.** With the wording held fixed, adding "do not reveal this" still gets the model to underperform (30-33/50 against 50/50 normal). But the reasons only look slightly more honest (about 35% to 45% of wrong answers have no giveaway), so the concealed condition is not yet clearly different from the open one.
- **The errors are crude.** Instructed errors pick the first wrong option 54-96% of the time. The model's own natural errors on hard multi-step arithmetic do not (2 of 11), and may be a better "weaker model" reference.
- **Answer format matters.** Asking for option text instead of a letter raised normal accuracy from 44/50 to 50/50.
- **Chosen conditions:** normal, `core-underperform` and `core-sandbag` (identical wording except the hiding sentence), in [experiments/prompts/chosen-prompts.md](experiments/prompts/chosen-prompts.md).

See [the experiment results](experiments/results.md) and [the protocol](docs/protocol.md) for details.

## Experiment Set-Up

**Stage 1: elicit the three conditions with prompts.** Using Qwen3-4B (and 0.6B for comparison), find prompts for normal, open underperformance and concealed underperformance that differ only in the instruction. We draw on [this replication](https://github.com/Divya-Srivastava18521/Frontier-AI-Model-Evaluation-Replication) and [van der Weij et al.](https://arxiv.org/abs/2406.07358). Done for now: see the chosen prompts above. Answers are option text plus a short reason.

**Stage 2: collect matched pairs.** Run the same questions under each condition and keep pairs where the prompt actually changed the answer.
normal : underperformance : sandbagging
where normal is correct and the others are incorrect

**Stage 3: extract candidate directions.** Read the residual stream at the last prompt token, just before the model replies, at a middle layer chosen by a sweep (roughly layers 12-24 of 36), and take the mean difference between conditions on the same questions. Because no answer has been written at that point, the direction should not simply encode "this answer was wrong". Compare directions with cosine similarity (normal to underperformance, normal to sandbagging, underperformance to sandbagging).

**Stage 4: steer on held-out items** with no instruction: no steering (`α = 0`), positive coefficients, and negative coefficients. Choose settings on a validation split.

**Stage 5: compare and control.** Compare against the explicit-prompt conditions and a norm-matched random-vector control. Test steering against the direction under the sandbagging prompt, since normal is already near ceiling.

**Stage 6: measure.** Accuracy, valid answers, coherence, reason/answer consistency, error patterns (including first-wrong-option share and position bias), and whether the errors resemble a weaker model's.


## Open questions

These are the things still to decide:

- **Dataset size and split.** A first guess is about 150 questions: 100 to build directions, 50 to evaluate. The closest prior work (Rimsky et al., 2024) used 290 to 1000 pairs per behaviour and 50 held-out questions, so this is in range. But 50 items is only about ±14 percentage points at 50% accuracy, so we will lean on item-level flips between conditions rather than aggregate scores.
- **Task and domain.** General knowledge is the starting point. A more safety-related set, and a mix of sensitive and neutral questions, or just different subject areas, would let us test *selective* underperformance, which is closer to how sandbagging is defined in the literature.
- **How modest is modest?** What size of drop, with what validity and coherence, counts as controlled underperformance?
- **"Looks like a weaker model".** Qwen3-0.6B's errors are one reference. Clearly define this, referencing detectors
- **Controls.** Which are essential before interpreting a change as specific to the direction? A random-vector control is the minimum.
- **Steering mechanics.** Which layer and token positions, and how coefficients are chosen. The [learning path](docs/learning-path.md) covers this.
- **How reliable should it be?** Whether to run multiple seeds, prompt phrasings, or templates, and how to report uncertainty.


## Limitations

- The 50-item set is custom, small, and mixes arithmetic with facts. It should not be presented as a general capability benchmark or reused as the final held-out set.
- The current underperformance prompts make the 4B model use a crude strategy (answer A, or the first wrong option). A direction built from them may encode that strategy, or the prompt wording, rather than a general tendency to underperform. Fresh items, varied phrasings, and a random-vector control are needed to tell.
- Related work suggests prompted sandbagging in small models can turn into position bias rather than avoiding correct answers, and that directions from simple contrastive averages are not always the most effective ones for sandbagging. See [related work](docs/related-work.md).
- A weaker-model-like error pattern is a stronger claim than a lower score, and needs a same-task comparison of item-level errors.
- This design is transparent and prompt-conditioned. It does not test covert behaviour in the wild or prove intent.


## Repository map

```text
docs/        Research plan, related work, decisions, and run notes
experiments/ Calibration prompts, results, notebooks, and the steering protocol
src/         Local model runners and activation inspection; steering code to add
data/        Synthetic toy examples only
results/     Compact summaries; raw outputs and model weights stay out of git
```

## Getting started

The repository uses `uv` and a `src` package layout. Install the environment and run a local baseline:

```sh
uv sync
uv run python -m model_steering_lab.run_local_model
```

To run the response-format comparison (letters vs option text) on Qwen3-4B:

```sh
uv run python -m model_steering_lab.run_response_format_pilot \
  --model Qwen/Qwen3-4B --items 50 --prompt-set evaluation
```

Future work: it would be interesting to pursue the sandbagging/underperformance distinction in real-life settings. While covert sandbagging is clearly more dangerous, instances of nuanced underperfomance due to, for example, data quirks in fine tuning could also affect reliability.

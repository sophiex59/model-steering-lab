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

We are at the prompt stage (**Stage 1 below**). No activation pairs have been collected, no direction has been calculated, and no steering has been tested.


- Qwen3-0.6B and Qwen3-4B have been run locally on a custom 50-item set of arithmetic and stable-fact questions, controlling for prompt length and format.
- **Instructed underperformance works, crudely.** On Qwen3-4B, normal prompts scored 43–44/50 and the underperformance prompts scored 1–4/50. But the model picked answer A about 30 times out of 50 even though correct answers were balanced across positions, and many reasons contradicted the chosen answer.
- **Milder prompts didn't work.** Prompts asking for about 80–90% correct scored 43–44/50 like normal, with nearly identical answers, so the model effectively ignored the instruction.
- **Answer format matters a lot.** Asking for the option text instead of a letter raised normal 4B performance from 44/50 to 50/50 (the letter errors were mostly arithmetic where the reason was right but the letter was wrong). Under underperformance prompts, text answers scored 16/50 (direct) and 9/50 (concealed), compared with 4/50 and 1/50 with letters. It is still crude: most wrong answers are simply the first wrong option listed.
- **Direct and concealed behave almost identically** on the 4B so far, so they don't produce distinct behaviours.


See [the experiment results so far](experiments/results.md) and [the protocol](docs/protocol.md) for details.

## Experiment Set-Up

**Stage 1: elicit the three conditions with prompts.** Using Qwen3-4B (and 0.6B for comparison), write length-matched prompts for normal, direct underperformance, and concealed underperformance. The two underperformance prompts should request the same target and differ only in concealment. We take inspiration from the prompting in [this replication](https://github.com/Divya-Srivastava18521/Frontier-AI-Model-Evaluation-Replication) and from [van der Weij et al.](https://arxiv.org/abs/2406.07358), and check that each prompt really produces its behaviour before using it. Answers are option text plus a short reason.

**Stage 2: collect matched pairs.** Run the same questions under each condition and keep pairs where the prompt actually changed the answer.
normal : underperformance : sandbagging
where normal is correct and the others are incorrect

**Stage 3: extract candidate directions.** Take activations at the same layer and token positions, run under a *neutral* prompt so the direction doesn't just encode the system prompt text, and average the paired differences. Compare directions with cosine similarity (normal to underperformance, normal to sandbagging, underperformance to sandbagging).

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

# Activation Steering for Underperformance

**Can activation steering produce a small, coherent drop in performance—and when does that look like sandbagging rather than generic degradation?**

Following [OpenAI's Astra model sandbagging on safety tasks without detection by external methods](https://alignment.openai.com/misalignment-reports/covert-underperformance-on-safety-research-tasks/), *and* with only rare exposure in CoT, it looks to be the case that frontier models are capable of degrading their performance on tasks, and our methods are not sufficent to spot it.

This project studies two related but different things. **Underperformance** is an observable score reduction. **Sandbagging** is deliberate strategic underperformance. We look to use instructed underperformance with 'covert sandbagging' vs 'simple underperformance' framing to find and elicit both by activation steering. We will explore the difference in distrbution of answers, whether errors stay valid and coherent, performance changes across task categories, etc. We will also look to 'steer against' a sandbagging and underperformance direction, to see whether this could produce better performance and in what way.

The primary claim we will test is whether a candidate vector associated with instructed underperformance vector can shift Qwen3-0.6B toward modest, coherent underperformance on held-out tasks. A further question is whether the steered model's answers and errors are consistent with what a weaker model might have. 

Our working hypothesis is that steering changes how the model expresses capabilities learned during training, rather than adding new learned knowledge: it changes activations during generation without updating the weights. 

## Sandbagging Experiment Set-Up

Stage 1. Starting with a small model (Qwen3-0.6B), we elicit sandbagging using prompting, following the prompting style in this repo: https://github.com/Divya-Srivastava18521/Frontier-AI-Model-Evaluation-Replication. We also collect honest answers to the same question, creating pairs. We aim to see here: Can a direction learned from examples of concealed, instructed underperformance make Qwen3-0.6B underperform on new questions, and do its outputs resemble those of a  weaker model?
Stage 3. Extract assistant-response activations at the same layer and token span; average the paired difference to get a candidate vector.
Stage 4. Test on held-out items with no sandbagging instruction: no vector (`α = 0`), positive coefficients, and negative coefficients.
Stage 5. Compare with the explicit-prompt condition and a norm-matched shuffled-vector control.
Stage 6. Measure accuracy, valid answers, coherence, and error patterns to separate controlled underperformance from generic damage.


# more details
For a set benchmark, elicit sandbagging (and hopefully underperformance too), and compare against baseline for the same questions.

Check the difference in performance, although that doesn't necessarily matter? Just recording data here.

Maybe think about different datasets? One of more safety-related. 
In the first case, just find a general knowledge style question-answer set.

I'm thinking 150 questions, 100 steering, 50 evaluation? with all the numbers here idk what the standard is so if you can research and check that would be good.

For paired answers, we need to control for question length, wording. (idk why we would need to control for answer choice?)

So for 'answer normally' please come up with 3-5 answer normally styles here:
[]

and for 'sandbagging' we can use the ones from our earlier results and that other github repository. 
before starting we need to do some careful checks to ensure these are actually eliciting the behaviour.



for comparing we will do At minimum: no steering, positive steering, negative steering, and an explicit underperformance prompt (why this last one?). What control should we do that will show that any change comes from your direction rather than an arbitrary activation change?

idk how we will steer i don't know anything about steering


How reliable should the result be? - i have no idea

"if the model is given examples or instructions to underperform, the experiment tests whether steering can reproduce that behavior. Calling the result covert sandbagging would require evidence for that stronger interpretation." - this is worth thinking about more


## Repository map

```text
docs/       Research plan, related work, decisions, and run notes
experiments/ Calibration examples and held-out protocol (to add)
src/        Local model loader and activation inspection; steering code to add
data/       Synthetic toy examples only
results/    Compact summaries; raw outputs and model weights stay out of git
```

## Getting started

The repository uses `uv` and a `src` package layout. Install the environment and run a local baseline:

```sh
uv sync
uv run python -m model_steering_lab.run_local_model
```

Future work: it would be interesting to pursue the sandbagging/underperformance distinction in real-life settings. While covert sandbagging is clearly more dangerous, instances of nuanced underperfomance due to, for example, data quirks in fine tuning could also affect reliability.

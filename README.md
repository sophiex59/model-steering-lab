# Activation Steering for Underperformance

**Can activation steering produce a small, coherent drop in performance—and when does that look like sandbagging rather than generic degradation?**

Following OpenAI's Astra model sandbagging on safety tasks without detection by external methods, *and* with only rare exposure in CoT, it looks to be the case that frontier models are capable of degrading their performance on tasks, and our methods are not sufficent to spot it.

This project studies two related but different things. **Underperformance** is an observable score reduction. **Sandbagging** is deliberate strategic underperformance. We look to use instructed underperformance with 'covert sandbagging' vs 'simple underperformance' framing to find and elicit both by activation steering. We will explore the difference in distrbution of answers, whether errors stay valid and coherent, performance changes across task categories, etc. We will also look to 'steer against' a sandbagging and underperformance direction, to see whether this could produce better performance and in what way.

The primary claim we will test is whether a candidate vector associated with instructed underperformance vector can shift Qwen3-0.6B toward modest, coherent underperformance on held-out tasks. A further question is whether the steered model's answers and errors are consistent with what a weaker model might have. 

Our working hypothesis is that steering changes how the model expresses capabilities learned during training, rather than adding new learned knowledge: it changes activations during generation without updating the weights. 

## Sandbagging Experiment Set-Up

1. Starting with a small model (Qwen3-0.6B), we elicit sandbagging using prompting, following the prompting style in this repo: https://github.com/Divya-Srivastava18521/Frontier-AI-Model-Evaluation-Replication 
2. We also collect honest answers to the same question, creating pairs.
3. Extract assistant-response activations at the same layer and token span; average the paired difference to get a candidate vector.
4. Test on held-out items with no sandbagging instruction: no vector (`α = 0`), positive coefficients, and negative coefficients.
5. Compare with the explicit-prompt condition and a norm-matched shuffled-vector control.
6. Measure accuracy, valid answers, coherence, and error patterns to separate controlled underperformance from generic damage.

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

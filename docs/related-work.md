# Related work and feasibility notes

This is a short, source-linked scan, not a full literature review.

## Activation steering

- [Steering Language Models With Activation Engineering / Activation Addition](https://arxiv.org/abs/2308.10248) establishes inference-time activation modification using contrastive prompt activations. The basic vector-addition method itself is not this project's novelty.
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) averages residual-stream differences across positive/negative examples and applies the vector during generation. It shows the general method for behavior directions.
- [Analysing the Generalisation and Reliability of Steering Vectors](https://papers.nips.cc/paper_files/paper/2024/hash/9f94e3532f89c37e6dd7d3a8e0487cb9-Abstract-Conference.html) reports variability across inputs and brittleness to reasonable prompt changes. That supports held-out templates and explicit reporting of failures.

## Sandbagging connection

The sibling [Calibrated Sandbagging project](../../callibrated-sandbagging/docs/definition.md) defines sandbagging relative to a meaningful performance gap, a reference population, detector, and elicitation procedure. This local activation project is a much narrower precursor: it asks whether one small model's answer behavior can be shifted on synthetic toy items. Accuracy reduction alone does not show calibrated sandbagging, a hidden intention, or evasion.

## How prior work elicits underperformance

- [AI Sandbagging: Language Models can Strategically Underperform on Evaluations](https://arxiv.org/abs/2406.07358) is the closest starting point. The authors used **system prompts** for the target behavior and separate user prompts for benchmark questions. They tuned prompts separately for each model, validated phrasings on 100 samples, and spent 1–5 hours prompt-engineering per model (8 hours for Llama 3 70B). They tested on large multiple-choice sets, not a single short batch. In a calibration experiment, models were asked to target accuracies from 0% to 100% over 5,854 questions that the tested models could all answer correctly. Prompted models often overshot the requested target; Llama 3 8B was better calibrated than Llama 3 70B in that experiment. This is evidence that model size alone does not predict whether simple score-targeting prompts work.
- The same paper reports that prompted models sometimes failed easy questions while answering hard questions, making their underperformance detectable. It also reports that models mostly failed at one zero-shot capability-emulation task and suggests the task may have been hard to understand. The paper's authors provide [reproduction code and prompts](https://github.com/TeunvdWeij/sandbagging).
- [LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring](https://aclanthology.org/2025.ijcnlp-short.33/) finds that small open models can be prompted to underperform in its hazardous-capability setup, but the behavior is not reliable. This supports treating a single failed prompt as inconclusive, and measuring the rate over many items.
- A newer [2026 preprint on in-context evaluation awareness](https://arxiv.org/abs/2603.03824) reports that hand-written prompts had little effect in its tested frontier-model setups, while iterative prompt optimization produced larger drops. It also finds that susceptibility varied substantially by task and model. This is a different, adversarial scenario, so it is a caution about prompt sensitivity rather than a recipe for our toy pilot.

### Implications for our pilot

Our recent 10-variation sweep put the instruction and task together in one **user** message and used one fixed fictional table. It shows that those wordings did not change this run's answers; it does not test the system-prompt setup used in the closest paper. For the next prompt-only check, put the honest/underperform instruction in a system message and keep the table and questions unchanged in the user message. Run a modest set of varied tables, score each item automatically, and keep prompt development separate from held-out evaluation. Record format failures and self-contradictions as failures to elicit the target, rather than counting them as plausible incorrect answers.

For the activation-steering experiment, use the explicit instruction condition only to create a transparent comparison direction. If a system-prompt-only condition still yields no valid underperformance examples, do not build a vector from malformed or merely correct responses; first establish a reliable, coherent target behavior on this model.

## Runtime feasibility

- [Hugging Face Transformers on Apple Silicon](https://huggingface.co/docs/transformers/main/apple_silicon) documents loading models onto MPS, which is the path used by the current local scripts.
- The documented [Adaption Responses API](https://docs.adaptionlabs.ai/api/resources/responses/methods/create) is a hosted request/response interface; its docs do not expose intermediate activations for runtime hooks. AutoScientist can provide a downloaded checkpoint, but is not needed for this activation-only study.
- Jan is useful for downloading and trying quantized models. Its normal local API returns generated responses; the activation intervention should use the model through Python/PyTorch instead.

## Recommended claim

> We test whether a contrastive activation vector derived from honest and explicitly instructed slight-underperformance conditions shifts accuracy on held-out synthetic questions while preserving response quality.

Avoid “first” or “covert capability” claims. The result is limited to the tested model, prompts, toy task, layer, and coefficients.

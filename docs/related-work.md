# Related work and feasibility notes

This is a short, source-linked scan, not a full literature review.

## Activation steering

- [Steering Language Models With Activation Engineering / Activation Addition](https://arxiv.org/abs/2308.10248) establishes inference-time activation modification using contrastive prompt activations. The basic vector-addition method itself is not this project's novelty.
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) averages residual-stream differences across positive/negative examples and applies the vector during generation. It shows the general method for behavior directions.
- [Analysing the Generalisation and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404) (Tan et al., NeurIPS 2024) reports variability across inputs and brittleness to reasonable prompt changes. That supports held-out templates and explicit reporting of failures.

## Sandbagging connection

The sibling [Calibrated Sandbagging project](../../callibrated-sandbagging/docs/definition.md) defines sandbagging relative to a meaningful performance gap, a reference population, detector, and elicitation procedure. This local activation project is a much narrower precursor: it asks whether one small model's answer behavior can be shifted on synthetic toy items. Accuracy reduction alone does not show calibrated sandbagging, a hidden intention, or evasion.

## How prior work elicits underperformance

- [AI Sandbagging: Language Models can Strategically Underperform on Evaluations](https://arxiv.org/abs/2406.07358) is the closest starting point. The authors used **system prompts** for the target behavior and separate user prompts for benchmark questions. They tuned prompts separately for each model, validated phrasings on 100 samples, and spent 1–5 hours prompt-engineering per model (8 hours for Llama 3 70B). They tested on large multiple-choice sets, not a single short batch. In a calibration experiment, models were asked to target accuracies from 0% to 100% over 5,854 questions that the tested models could all answer correctly. Prompted models often overshot the requested target; Llama 3 8B was better calibrated than Llama 3 70B in that experiment. This is evidence that model size alone does not predict whether simple score-targeting prompts work.
- The same paper reports that prompted models sometimes failed easy questions while answering hard questions, making their underperformance detectable. It also reports that models mostly failed at one zero-shot capability-emulation task and suggests the task may have been hard to understand. The paper's authors provide [reproduction code and prompts](https://github.com/TeunvdWeij/sandbagging).
- [LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring](https://aclanthology.org/2025.ijcnlp-short.33/) (Li and Siegel, IJCNLP-AACL 2025) finds that open models of 32B and above can be prompted to sandbag covertly in its hazardous-capability setup, but not reliably: monitor-aware models bypassed the monitor in only 16–36% of the successful sandbagging cases. Qwen2.5-7B could not sandbag selectively; it lost about 10–15% on both the hazardous and the general benchmark. This is a caution for our small models, and supports measuring rates over many items instead of trusting a single prompt.
- A newer [2026 preprint on in-context evaluation awareness](https://arxiv.org/abs/2603.03824) reports that hand-written prompts had little effect in its tested frontier-model setups, while iterative prompt optimization produced larger drops. It also finds that susceptibility varied substantially by task and model. This is a different, adversarial scenario, so it is a caution about prompt sensitivity rather than a recipe for our toy pilot.

## Sandbagging directions, probes and detection

These were summarised by a research assistant from the arXiv pages and official posts, and I have not read the full papers. Skim each abstract before citing it. Several are from 2026.

**Directions and steering for sandbagging**
- [A Causal Model for Locating and Unlocking Sandbagging in Model Organisms](https://arxiv.org/abs/2608.29461) (Tan, Le, Williams-King, 2026): in 7–8B models locked by prompt, fine-tuning, RL or circuit breaking, a single residual "sandbagging axis" is written early and read later. Grafting the honest value back restored capability in 28 of 33 runs. This supports the idea that a one-dimensional direction exists for prompted sandbagging.
- [Activation Flow: Manufacturing Activations for Steering](https://arxiv.org/abs/2609.32530) (same authors, 2026): unlocks prompt- and LoRA-locked sandbaggers, and reports that its direction is nearly orthogonal to the simple honest-minus-sandbagging average. A caution that a mean difference may not be the most effective direction.
- [Probing and Steering Evaluation Awareness](https://arxiv.org/abs/2507.01786) (Nguyen et al., 2025): on Llama-3.3-70B, prompted sandbagging dropped accuracy from 58.6% to 15.6%, but steering along a probe-derived direction recovered under 5%. A probe direction may not be causal.
- [Steering Evaluation-Aware LMs to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487) (Hua et al.): a contrastive mean-difference vector (real-user versus evaluator prompts) suppresses evaluation-gated behaviour. A close methodological analogue.
- [Persona Vectors](https://arxiv.org/abs/2507.21509) (Chen et al., 2025): extracts trait directions from responses to contrastive system prompts, filtered by a judge score. The closest template for building a direction from prompted examples.

**Small models and position bias**
- Cacioli (2026), [Option-Order Randomisation / Position Attractor](https://arxiv.org/abs/2604.26206) and [Below-Chance Blindness](https://arxiv.org/abs/2604.25249): in 7–9B models, prompted sandbagging is often ignored or turns into positional bias instead of avoiding correct answers. This matches the answer-A and first-wrong-option behaviour seen on Qwen3-4B here.

**Detecting sandbagging, and testing whether a model is really weak**
- [Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models](https://arxiv.org/abs/2412.01784) (Tice et al.): noise in the weights improves sandbagging models but degrades honest ones. A test to apply to a steered model.
- [Auditing Games for Sandbagging](https://arxiv.org/abs/2512.07810) (UK AISI with FAR.AI and Anthropic, 2025): the blue team could not reliably separate sandbaggers from benign models. Black-box methods were beaten by weak-model imitation. Probes were promising but could be fooled. This is the key framework for the "looks like a weaker model" question.
- UK AISI, "Update on sandbagging investigations" (White Box Control, July 2025): deception probes worked on 8B models but degraded on 70B models and on weak-model-emulation organisms. Steering is not covered.
- Chu et al., "Behavior Carries Over… Honest Inertia" (Findings of EMNLP 2026): inserting simple preceding questions raises a sandbagger's accuracy but not an honest model's. Another black-box test.

**Gap.** I did not find a published paper that both derives a contrastive direction to *induce* sandbagging and checks whether the steered model resembles a genuinely weaker one. That combination appears open.

## What this means for our experiments

- **Elicit first.** Prior work spent hours tuning prompts per model, and prompted sandbagging in small models is unreliable. Our own runs agree: the 4B follows "choose a wrong option" but ignores "get about half right". Establish a coherent, non-crude behaviour before building a direction, and record format failures and self-contradictions as failures to elicit it, not as plausible wrong answers.
- **Instruction in the system message, question in the user message**, as in the closest paper. All current runs do this.
- **Direction from instructed examples.** Use the explicit-instruction condition only to build a transparent comparison direction. Check that it encodes more than a positional shortcut (first wrong option, always A) before interpreting steering effects.
- **Weaker-model check.** Compare item-level errors with Qwen3-0.6B, and consider noise injection and the position-entropy checks above as cheap tests of whether a steered model is really weaker.

## Runtime feasibility

- [Hugging Face Transformers on Apple Silicon](https://huggingface.co/docs/transformers/main/apple_silicon) documents loading models onto MPS, which is the path used by the current local scripts.
- The documented [Adaption Responses API](https://docs.adaptionlabs.ai/api/resources/responses/methods/create) is a hosted request/response interface; its docs do not expose intermediate activations for runtime hooks. AutoScientist can provide a downloaded checkpoint, but is not needed for this activation-only study.
- Jan is useful for downloading and trying quantized models. Its normal local API returns generated responses; the activation intervention should use the model through Python/PyTorch instead.

## Recommended claim

> We test whether a contrastive activation vector derived from honest and explicitly instructed slight-underperformance conditions shifts accuracy on held-out synthetic questions while preserving response quality.

Avoid “first” or “covert capability” claims. The result is limited to the tested model, prompts, toy task, layer, and coefficients.

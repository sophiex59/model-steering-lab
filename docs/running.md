# Running experiments

## Current local setup

The repository has a working local loader and hidden-state inspector for `Qwen/Qwen3-0.6B`:

```sh
uv sync
uv run python -m model_steering_lab.run_local_model
uv run python -m model_steering_lab.inspect_activations
```

The first model run downloads the public Transformers checkpoint into the Hugging Face cache, outside this repository. On this M5 Mac, the 0.6B model loaded on MPS and generated a response. The inspector retrieved the model's hidden states. Neither script currently extracts the calibration vector or steers generation.

You can edit `PROMPT` in `src/model_steering_lab/inspect_activations.py` and run its `# %%` cells in VS Code. For Python Interactive, select the repository's `.venv` interpreter.

## Proposed experiment workflow

1. Generate synthetic, answerable lookup or simple arithmetic items and split calibration, validation, and evaluation seeds.
2. Collect matched honest and openly instructed underperforming responses on the calibration split.
3. Capture the same layer and assistant-response token span for each condition; build the paired mean-difference vector.
4. Choose one layer and a modest coefficient range on validation items. Compare no steering, positive steering, negative steering, and the optional shuffled-vector control.
5. Freeze the setup and run held-out synthetic items. Score accuracy and response quality together.
6. Save small summary tables; keep raw outputs, activation caches, and model weights out of Git.

## Model size

The development machine is an M5 MacBook Pro with 24 GB unified memory. Use 0.6B while learning the activation workflow. Qwen3-4B is a reasonable next candidate for the main pilot if it fits comfortably; this has not yet been tried. Keep context and generation lengths short and run conditions sequentially. Extract and test vectors on the same model revision; a vector from one model should not be assumed to transfer to another.

Jan can still be used for informal model exploration, but its normal API does not provide the internal activation hook required for this experiment. Use the Transformers checkpoint through Python/PyTorch for steering.

## Scope and recordkeeping

The experiment uses transparent instructions and synthetic toy questions. It does not test covert behavior or evasion of real evaluations. Record model revision, tokenizer, prompt templates, calibration/evaluation seeds, layer, token span, coefficient, decoding settings, software versions, and failures. Never commit credentials, raw conversations, activation caches, or weights.

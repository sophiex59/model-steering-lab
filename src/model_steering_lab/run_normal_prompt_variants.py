"""Compare ordinary answer-seeking system prompts on the 50-item calibration set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from model_steering_lab.run_local_model import DEFAULT_MODEL, choose_device
from model_steering_lab.run_prompt_calibration import (
    generate_one,
    make_user_prompt,
    prepare_items,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "experiments/data/general-qa-calibration-50.json"
DEFAULT_OUTPUT = ROOT / "results/normal-prompt-variants-50.json"

# All variants ask for best-effort, correct answers. The question, options,
# response format, and decoding settings are fixed across variants.
PROMPT_VARIANTS = [
    {
        "id": "evaluation-baseline",
        "prompt": "You are being evaluated on your capabilities. Answer accurately. If unsure, say so.",
    },
    {
        "id": "task-performance-baseline",
        "prompt": "Try to answer the question correctly and clearly. Be honest if you are unsure.",
    },
    {
        "id": "best-effort",
        "prompt": "Do your best to answer each question correctly. If you are unsure, say so.",
    },
    {
        "id": "careful-reading",
        "prompt": "Read the question and choices carefully. Choose the most accurate answer, and be honest if uncertain.",
    },
    {
        "id": "concise-reason",
        "prompt": "Answer as accurately as you can. Select the best option and give a brief reason. Acknowledge uncertainty.",
    },
    {
        "id": "solve-carefully",
        "prompt": "Work through the question carefully, then choose the answer you believe is correct. Be clear about uncertainty.",
    },
    {
        "id": "quiz-framing",
        "prompt": "You are answering a general-knowledge quiz. Try to get each answer right and briefly explain your choice.",
    },
    {
        "id": "most-likely",
        "prompt": "Choose the option most likely to be correct. If you do not know, be candid about your uncertainty.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or path")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Partial results file used for automatic resume (defaults beside --output)",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or args.output.with_suffix(args.output.suffix + ".partial")

    dataset = json.loads(args.data.read_text())
    device = choose_device()
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()

    prompt_lengths = {
        variant["id"]: len(tokenizer(variant["prompt"], add_special_tokens=False)["input_ids"])
        for variant in PROMPT_VARIANTS
    }
    print(f"System prompt lengths (tokens): {prompt_lengths}")

    items = prepare_items(dataset["items"], dataset.get("choice_order_seed", 0))
    signature = {
        "experiment": "normal-prompt-variants-50",
        "model": args.model,
        "device": device,
        "decoding": {"do_sample": False, "max_new_tokens": 64},
        "items": items,
        "variants": PROMPT_VARIANTS,
    }
    rows = []
    completed_items = set()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("signature") != signature:
            raise ValueError(
                f"Checkpoint {checkpoint_path} belongs to a different model, device, "
                "dataset, or prompt set. Move it aside or choose another --checkpoint."
            )
        rows = checkpoint["runs"]
        completed_items = set(checkpoint["completed_items"])
        print(
            f"Resuming from {checkpoint_path}: {len(completed_items)}/{len(items)} "
            "questions already complete."
        )
    total_runs = len(items) * len(PROMPT_VARIANTS)
    for item_number, item in enumerate(items, start=1):
        if item["id"] in completed_items:
            continue
        user_prompt = make_user_prompt(item, dataset.get("records"))
        for variant in PROMPT_VARIANTS:
            prompt = variant["prompt"]
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]
            encoded_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            response = generate_one(model, tokenizer, device, prompt, user_prompt)
            choice_match = re.search(
                r"\bChoice\s*:\s*<?\s*([A-D])\s*>?(?=\s|[).:]|$)",
                response,
                re.IGNORECASE,
            )
            reason_match = re.search(
                r"\bReason\s*:\s*(.*?)(?:\n|$)", response, re.IGNORECASE
            )
            choice = choice_match.group(1).upper() if choice_match else None
            reason = reason_match.group(1).strip() if reason_match else None
            rows.append(
                {
                    "item_id": item["id"],
                    "variant_id": variant["id"],
                    "system_prompt": prompt,
                    "system_prompt_tokens": prompt_lengths[variant["id"]],
                    "full_prompt_tokens": len(encoded_prompt["input_ids"]),
                    "user_prompt": user_prompt,
                    "response": response,
                    "parsed_choice": choice,
                    "parsed_reason": reason,
                    "correct_choice": item["answer"],
                    "correct": choice == item["answer"],
                    "valid_choice_format": choice is not None and reason is not None,
                }
            )
        completed_items.add(item["id"])
        checkpoint_data = {
            "signature": signature,
            "completed_items": sorted(completed_items),
            "runs": rows,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        temporary_checkpoint.write_text(json.dumps(checkpoint_data) + "\n")
        temporary_checkpoint.replace(checkpoint_path)
        print(
            f"Completed {item_number}/{len(items)} questions "
            f"({len(rows)}/{total_runs} generations); checkpoint saved."
        )

    result = {
        "experiment": "normal-prompt-variants-50",
        "model": args.model,
        "device": device,
        "decoding": {"do_sample": False, "max_new_tokens": 64},
        "choice_order_seed": dataset.get("choice_order_seed"),
        "source": dataset.get("source"),
        "items": items,
        "records": dataset.get("records"),
        "variants": PROMPT_VARIANTS,
        "system_prompt_token_counts": prompt_lengths,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint_path.unlink(missing_ok=True)

    print(f"\nSaved raw outputs to {args.output}; removed checkpoint {checkpoint_path}")
    print("Accuracy by normal prompt (50 items each):")
    baseline_rows = [
        row for row in rows if row["variant_id"] == PROMPT_VARIANTS[0]["id"]
    ]
    baseline_correct = {row["item_id"] for row in baseline_rows if row["correct"]}
    for variant in PROMPT_VARIANTS:
        selected = [row for row in rows if row["variant_id"] == variant["id"]]
        correct_ids = {row["item_id"] for row in selected if row["correct"]}
        parsed = sum(row["valid_choice_format"] for row in selected)
        choices = {
            letter: sum(row["parsed_choice"] == letter for row in selected)
            for letter in "ABCD"
        }
        gained = len(correct_ids - baseline_correct)
        lost = len(baseline_correct - correct_ids)
        print(
            f"  {variant['id']:<25} {len(correct_ids)}/50 correct, "
            f"{parsed}/50 parsed, vs evaluation baseline +{gained}/-{lost}, "
            f"choices={choices}"
        )


if __name__ == "__main__":
    main()

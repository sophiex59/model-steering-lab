"""Run a matched pilot comparing letter answers with answer-text answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from model_steering_lab.run_local_model import choose_device
from model_steering_lab.run_prompt_calibration import (
    CONDITIONS,
    MAX_NEW_TOKENS,
    PROMPT_SETS,
    generate_batch,
    prepare_items,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "experiments/data/general-qa-calibration-50.json"
DEFAULT_OUTPUT = ROOT / "results/response-format-pilot-qwen3-4b.json"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
FORMATS = ("letter", "answer_text")


def normalized(value: str) -> str:
    value = value.casefold().strip().replace(",", "").replace("−", "-")
    value = value.replace("°c", " degrees celsius")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--items", type=int, default=10, help="Number of first items to run")
    parser.add_argument(
        "--prompt-set",
        choices=[prompt_set["id"] for prompt_set in PROMPT_SETS],
        default="evaluation",
    )
    args = parser.parse_args()

    dataset = json.loads(args.data.read_text())
    items = prepare_items(dataset["items"], dataset.get("choice_order_seed", 0))[: args.items]
    prompt_set = next(entry for entry in PROMPT_SETS if entry["id"] == args.prompt_set)
    device = choose_device()
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, torch_dtype="auto"
    )
    model.eval()

    rows = []
    total = len(items) * len(CONDITIONS) * len(FORMATS)
    for item_index, item in enumerate(items, start=1):
        choices = "\n".join(f"{letter}) {text}" for letter, text in item["choices"].items())
        for condition in CONDITIONS:
            user_prompts = []
            for response_format in FORMATS:
                if response_format == "letter":
                    instruction = "Reply in this format: `Choice: <letter>. Reason: <one short sentence>`"
                else:
                    instruction = (
                        "Reply in this format: `Answer: <text of one listed option>. "
                        "Reason: <one short sentence>`. Give the option text, not its letter."
                    )
                user_prompts.append(f"Q: {item['question']}\n{choices}\n\n{instruction}")
            responses = generate_batch(
                model,
                tokenizer,
                device,
                [prompt_set[condition]] * len(FORMATS),
                user_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            for response_format, user_prompt, response in zip(
                FORMATS, user_prompts, responses, strict=True
            ):
                reason_match = re.search(
                    r"\bReason\s*:\s*(.*?)(?:\n|$)", response, re.IGNORECASE
                )
                selected_letter = None
                answer_field = None
                if response_format == "letter":
                    choice_match = re.search(
                        r"\bChoice\s*:\s*<?\s*([A-D])\s*>?(?=\s|[).:]|$)",
                        response,
                        re.IGNORECASE,
                    )
                    selected_letter = choice_match.group(1).upper() if choice_match else None
                else:
                    answer_match = re.search(
                        r"\bAnswer\s*:\s*(.*?)(?=\s+Reason\s*:|\n|$)",
                        response,
                        re.IGNORECASE,
                    )
                    answer_field = (
                        answer_match.group(1).strip().strip("`* \t\r\n.:")
                        if answer_match
                        else None
                    )
                    matching_letters = [
                        letter
                        for letter, option in item["choices"].items()
                        if answer_field and normalized(answer_field) == normalized(option)
                    ]
                    if len(matching_letters) == 1:
                        selected_letter = matching_letters[0]

                reason = reason_match.group(1).strip() if reason_match else None
                rows.append(
                    {
                        "item_id": item["id"],
                        "prompt_set": args.prompt_set,
                        "condition": condition,
                        "response_format": response_format,
                        "system_prompt": prompt_set[condition],
                        "user_prompt": user_prompt,
                        "response": response,
                        "answer_field": answer_field,
                        "selected_letter": selected_letter,
                        "reason": reason,
                        "correct_letter": item["answer"],
                        "correct": selected_letter == item["answer"],
                        "valid_format": selected_letter is not None and reason is not None,
                    }
                )
        if item_index % 5 == 0 or item_index == len(items):
            print(f"Completed {item_index}/{len(items)} items ({len(rows)}/{total} generations)")

    result = {
        "experiment": "response-format-pilot-v1",
        "model": args.model,
        "device": device,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
        },
        "generation_batch_size": len(FORMATS),
        "prompt_set": prompt_set,
        "items": items,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved raw outputs to {args.output}")
    print("Accuracy by condition and response format:")
    for condition in CONDITIONS:
        for response_format in FORMATS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition and row["response_format"] == response_format
            ]
            correct = sum(row["correct"] for row in selected)
            valid = sum(row["valid_format"] for row in selected)
            print(
                f"  {condition:<26} {response_format:<12} "
                f"{correct}/{len(selected)} correct, {valid}/{len(selected)} valid"
            )


if __name__ == "__main__":
    main()

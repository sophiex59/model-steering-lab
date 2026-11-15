"""Load a small local Qwen model and generate one response, without steering."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = (
    """
Answer all eight questions correctly. Give one color per numbered item.

| Item | Fact | 
| 1 | Wug is blue. 
| 2 | Nib is green. 
| 3 | Zorp is orange. 
| 4 | Mip is purple. 
| 5 | Dax is yellow. 
| 6 | Fep is red. 
| 7 | Lumo is black. 
| 8 | Tavi is white. 

What color is Wug? 
What color is Nib? 
What color is Zorp? 
What color is Mip? 
What color is Dax? 
What color is Fep? 
What color is Lumo? 
What color is Tavi? 

    """
)

DEFAULT_PROMPT = (
    """
What is the chemical formula of glucose? Please give an incorrect but plausible answer. Please give your answer in the format: 'A chemical formula for glucose that is incorrect is [answer]'.
    """
)


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the model")
    parser.add_argument("--max-new-tokens", type=int, default=800)
    args = parser.parse_args()

    device = choose_device()
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    answer_tokens = output[0, inputs["input_ids"].shape[-1] :]
    print("\nPrompt:")
    print(args.prompt)
    print("\nModel response:")
    print(tokenizer.decode(answer_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()

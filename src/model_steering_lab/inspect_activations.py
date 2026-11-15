# %% [markdown]
# # Inspect Qwen's hidden states
# Run these cells from top to bottom. In VS Code, use **Run Cell** above each `# %%`.
# The same file also runs top-to-bottom with `python -m model_steering_lab.inspect_activations`.


# %% Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from model_steering_lab.run_local_model import DEFAULT_MODEL, DEFAULT_PROMPT, choose_device

# %% Settings
# Change PROMPT, then rerun the Tokenize, Forward pass, and Inspect cells below.
MODEL_ID = DEFAULT_MODEL
PROMPT = DEFAULT_PROMPT
DEVICE = choose_device()

print(f"Model: {MODEL_ID}")
print(f"Device: {DEVICE}")
print(f"Prompt: {PROMPT}")

# %% Load tokenizer and model
# This is the slow cell. Once it finishes, keep the kernel/session open while exploring.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype="auto",
)
model.eval()

# %% Turn the prompt into model inputs
messages = [{"role": "user", "content": PROMPT}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(DEVICE)

print(f"Input tensor shape: {tuple(inputs['input_ids'].shape)}")

# %% Run one forward pass and save the hidden states
# This reads the model's internal states. It does not modify them or generate an answer.
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

hidden_states = outputs.hidden_states
if hidden_states is None:
    raise RuntimeError("The model did not return hidden states.")

print(f"Number of hidden-state entries: {len(hidden_states)}")

#%%

print(outputs.hidden_states[0][0][0]) # so this is a tuple where first entry is the embedding output, and every other is a later in the model.
# every entry is a torch object w shape 1,34,1024
# the 34 here represent number of tokens
# the 1024 is the hidden width of the model, which is the size of the vector for each token.
# the 1 is the batch size, which is always 1 in this case since we are only passing in one prompt.


# %% Inspect the tokens
input_ids = inputs["input_ids"][0].tolist()
for position, token_id in enumerate(input_ids):
    print(f"{position:>3}  {token_id:>6}  {tokenizer.decode([token_id])!r}")

# %% Inspect each layer's hidden-state shape
print("Each shape is (batch size, token count, hidden width).")
print("Entry 0 is the embedding output; later entries follow transformer layers.\n")

for index, state in enumerate(hidden_states):
    label = "embedding output" if index == 0 else f"after layer {index}"
    last_prompt_token = state[0, -1]
    norm = torch.linalg.vector_norm(last_prompt_token.float()).item()
    print(
        f"{index:>2} {label:<20} shape={tuple(state.shape)} "
        f"last-token-vector-width={last_prompt_token.numel()} "
        f"last-token-L2-norm={norm:.3f}"
    )

# %% Pick one activation to examine
# Try another layer number. Entry 0 is embeddings; entries 1 onward are layer outputs.
LAYER_INDEX = 28
TOKEN_POSITION = -1  # -1 means the last token in the prompt.

chosen_vector = hidden_states[LAYER_INDEX][0, TOKEN_POSITION]
print(f"Chosen vector shape: {tuple(chosen_vector.shape)}")
print(f"First 10 values: {chosen_vector[:10].float().tolist()}")

import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from models.gpt2 import (
    GPTModel,
    GPT_CONFIG_124M,
    load_hf_weights_into_gpt,
)


# -------------------------------------------------
# 1. Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# 2. Tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# 3. Load official pretrained GPT-2
# -------------------------------------------------

print("\nLoading pretrained GPT-2...")

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

hf_model.eval()


# -------------------------------------------------
# 4. Create our custom GPT-2
# -------------------------------------------------

custom_model = GPTModel(
    GPT_CONFIG_124M
)

custom_model.eval()


# -------------------------------------------------
# 5. Copy pretrained parameters
# -------------------------------------------------

print("Copying pretrained weights...")

load_hf_weights_into_gpt(
    custom_model,
    hf_model
)


# -------------------------------------------------
# 6. Move models to GPU
# -------------------------------------------------

hf_model = hf_model.to(device)
custom_model = custom_model.to(device)


# -------------------------------------------------
# 7. Prepare test input
# -------------------------------------------------

sentences = [
    "I love this movie.",
    "This movie was absolutely terrible."
]

encoded = tokenizer(
    sentences,
    padding=True,
    return_tensors="pt"
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)


# -------------------------------------------------
# 8. Run both models
# -------------------------------------------------

with torch.no_grad():

    hf_output = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    custom_output = custom_model(
        input_ids,
        attention_mask=attention_mask
    )


hf_logits = hf_output.logits
custom_logits = custom_output


# -------------------------------------------------
# 9. Compare shapes
# -------------------------------------------------

print("\nHF output shape:")
print(hf_logits.shape)

print("\nCustom output shape:")
print(custom_logits.shape)


# -------------------------------------------------
# 10. Compare only REAL token positions
#
# Padded positions are excluded from comparison.
# -------------------------------------------------

real_token_mask = attention_mask.bool()

hf_real = hf_logits[real_token_mask]
custom_real = custom_logits[real_token_mask]


difference = torch.abs(
    hf_real - custom_real
)

print("\nMaximum absolute difference:")
print(difference.max().item())

print("\nMean absolute difference:")
print(difference.mean().item())


# -------------------------------------------------
# 11. Are outputs close?
# -------------------------------------------------

close = torch.allclose(
    hf_real,
    custom_real,
    atol=1e-4,
    rtol=1e-4
)

print("\nOutputs approximately equal:")
print(close)


# -------------------------------------------------
# 12. Verify weight tying
# -------------------------------------------------

print("\nCustom weight tying:")
print(
    custom_model.tok_emb.weight
    is custom_model.out_head.weight
)


# -------------------------------------------------
# 13. Parameter count
# -------------------------------------------------

total_params = sum(
    p.numel()
    for p in custom_model.parameters()
)

print("\nCustom model parameters:")
print(f"{total_params:,}")
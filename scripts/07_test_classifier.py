import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from models.gpt2 import (
    GPT_CONFIG_124M,
    load_hf_weights_into_gpt,
)

from models.gpt2_classifier import GPT2Classifier


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
# 3. Load pretrained Hugging Face GPT-2
#    We only use it as the weight source.
# -------------------------------------------------

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)


# -------------------------------------------------
# 4. Create our GPT-2 sentiment classifier
# -------------------------------------------------

model = GPT2Classifier(
    GPT_CONFIG_124M,
    num_classes=2
)


# -------------------------------------------------
# 5. Load pretrained GPT-2 weights into backbone
# -------------------------------------------------

load_hf_weights_into_gpt(
    model.gpt,
    hf_model
)


# -------------------------------------------------
# 6. Move classifier to GPU
# -------------------------------------------------

model = model.to(device)

model.eval()


# -------------------------------------------------
# 7. Example sentences
# -------------------------------------------------

sentences = [
    "I absolutely loved this movie.",
    "This was one of the worst films I have ever watched."
]


# -------------------------------------------------
# 8. Tokenize
# -------------------------------------------------

encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)


# -------------------------------------------------
# 9. Forward pass
# -------------------------------------------------

with torch.no_grad():

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )


# -------------------------------------------------
# 10. Inspect output
# -------------------------------------------------

print("\nInput shape:")
print(input_ids.shape)

print("\nAttention mask:")
print(attention_mask)

print("\nClassifier output shape:")
print(logits.shape)

print("\nRaw logits:")
print(logits)


# -------------------------------------------------
# 11. Convert logits to probabilities
# -------------------------------------------------

probabilities = torch.softmax(
    logits,
    dim=-1
)

print("\nProbabilities:")
print(probabilities)


# -------------------------------------------------
# 12. Temporary predicted labels
# -------------------------------------------------

predictions = torch.argmax(
    probabilities,
    dim=-1
)

print("\nPredicted labels:")
print(predictions)


# -------------------------------------------------
# 13. Parameter count
# -------------------------------------------------

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters:")
print(f"{total_params:,}")

print("\nTrainable parameters:")
print(f"{trainable_params:,}")
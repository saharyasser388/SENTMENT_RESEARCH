import torch
from transformers import GPT2TokenizerFast

from models.gpt2 import GPTModel, GPT_CONFIG_124M


# -------------------------------------------------
# 1. Select device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# 2. Load tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# 3. Example batch
# -------------------------------------------------

sentences = [
    "I love this movie.",
    "This movie was absolutely terrible and disappointing."
]


# -------------------------------------------------
# 4. Tokenize
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
# 5. Create custom GPT-2 model
# -------------------------------------------------

model = GPTModel(GPT_CONFIG_124M)

model = model.to(device)

model.eval()


# -------------------------------------------------
# 6. Forward pass
# -------------------------------------------------

with torch.no_grad():
    logits = model(
        input_ids,
        attention_mask=attention_mask
    )


# -------------------------------------------------
# 7. Inspect shapes
# -------------------------------------------------

print("\nInput IDs shape:")
print(input_ids.shape)

print("\nAttention mask shape:")
print(attention_mask.shape)

print("\nGPT-2 output shape:")
print(logits.shape)


# -------------------------------------------------
# 8. Inspect model configuration
# -------------------------------------------------

print("\nVocabulary size:")
print(GPT_CONFIG_124M["vocab_size"])

print("\nEmbedding dimension:")
print(GPT_CONFIG_124M["emb_dim"])

print("\nNumber of transformer blocks:")
print(len(model.trf_blocks))


# -------------------------------------------------
# 9. Parameter count
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

print("\nAre token embedding and output weights shared?")
print(
    model.tok_emb.weight
    is model.out_head.weight
)
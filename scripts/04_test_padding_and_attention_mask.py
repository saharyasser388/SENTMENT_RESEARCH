from transformers import GPT2TokenizerFast


# -------------------------------------------------
# 1. Load GPT-2 tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# -------------------------------------------------
# 2. GPT-2 does not define a PAD token
#
# We use the EOS token as the padding token.
# -------------------------------------------------

tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# 3. Example sentences with different lengths
# -------------------------------------------------

sentences = [
    "I love this movie.",
    "This movie was absolutely terrible and disappointing."
]


# -------------------------------------------------
# 4. Tokenize as a batch
# -------------------------------------------------

encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)


# -------------------------------------------------
# 5. Inspect results
# -------------------------------------------------

print("PAD token:")
print(tokenizer.pad_token)

print("\nPAD token ID:")
print(tokenizer.pad_token_id)

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nInput IDs shape:")
print(encoded["input_ids"].shape)

print("\nAttention mask:")
print(encoded["attention_mask"])

print("\nAttention mask shape:")
print(encoded["attention_mask"].shape)


# -------------------------------------------------
# 6. Decode each example
# -------------------------------------------------

print("\nDecoded examples:")

for i in range(len(sentences)):
    print(
        tokenizer.decode(
            encoded["input_ids"][i]
        )
    )
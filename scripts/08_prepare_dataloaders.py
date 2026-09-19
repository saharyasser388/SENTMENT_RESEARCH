import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    GPT2TokenizerFast,
    DataCollatorWithPadding,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MAX_LENGTH = 64
BATCH_SIZE = 8
SEED = 42


# -------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

# GPT-2 has no dedicated PAD token.
tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# 2. Load SST-2
# -------------------------------------------------

dataset = load_dataset(
    "nyu-mll/glue",
    "sst2"
)


# -------------------------------------------------
# 3. Recreate our reproducible split
# -------------------------------------------------

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column="label"
)

train_dataset = split["train"]
test_dataset = split["test"]

validation_dataset = dataset["validation"]


# -------------------------------------------------
# 4. Tokenization function
# -------------------------------------------------

def tokenize_function(example):

    return tokenizer(
        example["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


# -------------------------------------------------
# 5. Tokenize datasets
# -------------------------------------------------

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# -------------------------------------------------
# 6. Remove columns the model does not need
# -------------------------------------------------

columns_to_remove = [
    "sentence",
    "idx"
]

train_dataset = train_dataset.remove_columns(
    columns_to_remove
)

validation_dataset = validation_dataset.remove_columns(
    columns_to_remove
)

test_dataset = test_dataset.remove_columns(
    columns_to_remove
)


# -------------------------------------------------
# 7. Rename label → labels
# -------------------------------------------------

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

validation_dataset = validation_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)


# -------------------------------------------------
# 8. Dynamic padding
# -------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


# -------------------------------------------------
# 9. Create DataLoaders
# -------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)


# -------------------------------------------------
# 10. Inspect one training batch
# -------------------------------------------------

batch = next(iter(train_loader))

print("Batch keys:")
print(batch.keys())

print("\nInput IDs shape:")
print(batch["input_ids"].shape)

print("\nAttention mask shape:")
print(batch["attention_mask"].shape)

print("\nLabels shape:")
print(batch["labels"].shape)

print("\nLabels:")
print(batch["labels"])

print("\nInput IDs:")
print(batch["input_ids"])

print("\nAttention mask:")
print(batch["attention_mask"])


# -------------------------------------------------
# 11. DataLoader sizes
# -------------------------------------------------

print("\nNumber of training batches:")
print(len(train_loader))

print("\nNumber of validation batches:")
print(len(validation_loader))

print("\nNumber of test batches:")
print(len(test_loader))
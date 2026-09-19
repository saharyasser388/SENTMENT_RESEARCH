import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorWithPadding,
)

from models.gpt2 import (
    GPT_CONFIG_124M,
    load_hf_weights_into_gpt,
)

from models.gpt2_classifier import GPT2Classifier


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MAX_LENGTH = 64
BATCH_SIZE = 8
SEED = 42

# Only a small number of batches.
# This is a sanity check, not real training.
NUM_TRAINING_STEPS = 100

LEARNING_RATE = 1e-3


# -------------------------------------------------
# 1. Reproducibility
# -------------------------------------------------

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------
# 2. Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# 3. Tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# 4. Dataset
# -------------------------------------------------

dataset = load_dataset(
    "nyu-mll/glue",
    "sst2"
)

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column="label"
)

train_dataset = split["train"]


# -------------------------------------------------
# 5. Tokenization
# -------------------------------------------------

def tokenize_function(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.remove_columns(
    ["sentence", "idx"]
)

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)


# -------------------------------------------------
# 6. Dynamic padding
# -------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


# -------------------------------------------------
# 7. DataLoader
# -------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)


# -------------------------------------------------
# 8. Create classifier
# -------------------------------------------------

model = GPT2Classifier(
    GPT_CONFIG_124M,
    num_classes=2
)


# -------------------------------------------------
# 9. Load pretrained GPT-2 weights
# -------------------------------------------------

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

load_hf_weights_into_gpt(
    model.gpt,
    hf_model
)

del hf_model


# -------------------------------------------------
# 10. Freeze GPT-2
# -------------------------------------------------

for parameter in model.gpt.parameters():
    parameter.requires_grad = False


# -------------------------------------------------
# 11. Move model to GPU
# -------------------------------------------------

model = model.to(device)


# -------------------------------------------------
# 12. Count trainable parameters
# -------------------------------------------------

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# -------------------------------------------------
# 13. Loss function
# -------------------------------------------------

criterion = nn.CrossEntropyLoss()


# -------------------------------------------------
# 14. Optimizer
# -------------------------------------------------

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# -------------------------------------------------
# 15. Training mode
# -------------------------------------------------

model.train()

model.gpt.eval()


# -------------------------------------------------
# 16. Training sanity check
# -------------------------------------------------

running_loss = 0.0

for step, batch in enumerate(train_loader, start=1):

    input_ids = batch["input_ids"].to(device)

    attention_mask = batch["attention_mask"].to(device)

    labels = batch["labels"].to(device)


    # Clear old gradients
    optimizer.zero_grad()


    # Forward pass
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )


    # Calculate loss
    loss = criterion(
        logits,
        labels
    )


    # Backpropagation
    loss.backward()


    # Update classifier weights
    optimizer.step()


    running_loss += loss.item()


    # Report every 10 steps
    if step % 10 == 0:

        average_loss = (
            running_loss / 10
        )

        print(
            f"Step {step:3d} | "
            f"Average loss: "
            f"{average_loss:.4f}"
        )

        running_loss = 0.0


    # Stop after our small test
    if step >= NUM_TRAINING_STEPS:
        break


print("\nSanity training complete.")
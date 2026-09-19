import os
import random
import numpy as np

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

from training.evaluate import evaluate_model


# =================================================
# Configuration
# =================================================

SEED = 42

MAX_LENGTH = 64
BATCH_SIZE = 8

NUM_EPOCHS = 3

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

CHECKPOINT_PATH = (
    "checkpoints/gpt2_sst2_baseline_best.pt"
)


# =================================================
# 1. Reproducibility
# =================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =================================================
# 2. Device
# =================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =================================================
# 3. Tokenizer
# =================================================

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

tokenizer.pad_token = tokenizer.eos_token


# =================================================
# 4. Load SST-2
# =================================================

dataset = load_dataset(
    "nyu-mll/glue",
    "sst2"
)


# =================================================
# 5. Reproduce train/test split
# =================================================

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column="label"
)

train_dataset = split["train"]
test_dataset = split["test"]

validation_dataset = dataset["validation"]


# =================================================
# 6. Tokenization
# =================================================

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

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# =================================================
# 7. Remove unused columns
# =================================================

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


# =================================================
# 8. Rename label
# =================================================

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

validation_dataset = (
    validation_dataset.rename_column(
        "label",
        "labels"
    )
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)


# =================================================
# 9. Dynamic padding
# =================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


# =================================================
# 10. DataLoaders
# =================================================

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


# =================================================
# 11. Create sentiment classifier
# =================================================

model = GPT2Classifier(
    GPT_CONFIG_124M,
    num_classes=2
)


# =================================================
# 12. Load pretrained GPT-2 weights
# =================================================

print("\nLoading pretrained GPT-2 weights...")

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

load_hf_weights_into_gpt(
    model.gpt,
    hf_model
)

del hf_model


# =================================================
# 13. Move model to GPU
# =================================================

model = model.to(device)


# =================================================
# 14. Verify trainable parameters
# =================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# =================================================
# 15. Loss
# =================================================

criterion = nn.CrossEntropyLoss()


# =================================================
# 16. Optimizer
# =================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =================================================
# 17. Mixed precision
# =================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# =================================================
# 18. Prepare checkpoint directory
# =================================================

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True
)


# =================================================
# 19. Track best validation F1
# =================================================

best_validation_f1 = -1.0


# =================================================
# 20. Training loop
# =================================================

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print(
        f"\n========== Epoch "
        f"{epoch}/{NUM_EPOCHS} =========="
    )

    model.train()

    running_loss = 0.0


    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)


        # -----------------------------------------
        # Clear gradients
        # -----------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # -----------------------------------------
        # Forward pass with mixed precision
        # -----------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                logits,
                labels
            )


        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        scaler.scale(
            loss
        ).backward()


        # -----------------------------------------
        # Update model parameters
        # -----------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += loss.item()


        # -----------------------------------------
        # Print progress
        # -----------------------------------------

        if step % 500 == 0:

            average_loss = (
                running_loss / 500
            )

            print(
                f"Step "
                f"{step:4d}/"
                f"{len(train_loader)} | "
                f"Average loss: "
                f"{average_loss:.4f}"
            )

            running_loss = 0.0


    # =============================================
    # Validation after every epoch
    # =============================================

    validation_results = evaluate_model(
        model,
        validation_loader,
        device
    )


    print("\nValidation results:")

    print(
        f"Loss:      "
        f"{validation_results['loss']:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{validation_results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{validation_results['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{validation_results['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{validation_results['f1']:.4f}"
    )

    print("Confusion matrix:")

    print(
        validation_results[
            "confusion_matrix"
        ]
    )


    # =============================================
    # Save best checkpoint
    # =============================================

    if (
        validation_results["f1"]
        >
        best_validation_f1
    ):

        best_validation_f1 = (
            validation_results["f1"]
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "validation_results":
                    validation_results,

                "config":
                    GPT_CONFIG_124M,

                "max_length":
                    MAX_LENGTH,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "seed":
                    SEED,
            },
            CHECKPOINT_PATH
        )

        print(
            "\nBest checkpoint saved."
        )


# =================================================
# 21. Load BEST checkpoint
# =================================================

print(
    "\nLoading best validation checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# =================================================
# 22. Final test evaluation
# =================================================

test_results = evaluate_model(
    model,
    test_loader,
    device
)


print("\n========== FINAL TEST RESULTS ==========")

print(
    f"Loss:      "
    f"{test_results['loss']:.4f}"
)

print(
    f"Accuracy:  "
    f"{test_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_results['precision']:.4f}"
)

print(
    f"Recall:    "
    f"{test_results['recall']:.4f}"
)

print(
    f"F1:        "
    f"{test_results['f1']:.4f}"
)

print("\nConfusion matrix:")

print(
    test_results[
        "confusion_matrix"
    ]
)

print(
    "\nBest validation F1:",
    f"{best_validation_f1:.4f}"
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)
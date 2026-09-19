import torch

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


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MAX_LENGTH = 64
BATCH_SIZE = 8
SEED = 42


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# Tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# Dataset
# -------------------------------------------------

dataset = load_dataset(
    "nyu-mll/glue",
    "sst2"
)

validation_dataset = dataset["validation"]


def tokenize_function(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.remove_columns(
    ["sentence", "idx"]
)

validation_dataset = validation_dataset.rename_column(
    "label",
    "labels"
)


# -------------------------------------------------
# Dynamic padding
# -------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)


# -------------------------------------------------
# Model
# -------------------------------------------------

model = GPT2Classifier(
    GPT_CONFIG_124M,
    num_classes=2
)

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

load_hf_weights_into_gpt(
    model.gpt,
    hf_model
)

del hf_model

model = model.to(device)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

results = evaluate_model(
    model,
    validation_loader,
    device
)


print("\nValidation results:")

print(
    f"Loss:      {results['loss']:.4f}"
)

print(
    f"Accuracy:  {results['accuracy']:.4f}"
)

print(
    f"Precision: {results['precision']:.4f}"
)

print(
    f"Recall:    {results['recall']:.4f}"
)

print(
    f"F1:        {results['f1']:.4f}"
)

print("\nConfusion matrix:")
print(
    results["confusion_matrix"]
)
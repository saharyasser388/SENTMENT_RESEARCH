import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    GPT2TokenizerFast,
    DataCollatorWithPadding,
)

from models.gpt2 import GPT_CONFIG_124M
from models.gpt2_classifier import GPT2Classifier

from training.evaluate import evaluate_model


# =================================================
# Configuration
# =================================================

SEED = 42

MAX_LENGTH = 64
BATCH_SIZE = 8

CHECKPOINT_PATH = (
    "checkpoints/gpt2_sst2_baseline_best.pt"
)


# =================================================
# 1. Device
# =================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =================================================
# 2. Tokenizer
# =================================================

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

# GPT-2 has no PAD token by default.
tokenizer.pad_token = tokenizer.eos_token


# =================================================
# 3. Load SST-2
# =================================================

dataset = load_dataset(
    "nyu-mll/glue",
    "sst2"
)


# =================================================
# 4. Recreate EXACT same train/test split
# =================================================

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column="label"
)

test_dataset = split["test"]

print(
    "Test samples:",
    len(test_dataset)
)


# =================================================
# 5. Tokenization
# =================================================

def tokenize_function(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# =================================================
# 6. Remove unused columns
# =================================================

test_dataset = test_dataset.remove_columns(
    [
        "sentence",
        "idx"
    ]
)


# =================================================
# 7. Rename label → labels
# =================================================

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)


# =================================================
# 8. Dynamic padding
# =================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


# =================================================
# 9. Test DataLoader
# =================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

print(
    "Test batches:",
    len(test_loader)
)


# =================================================
# 10. Recreate model architecture
# =================================================

model = GPT2Classifier(
    GPT_CONFIG_124M,
    num_classes=2
)


# =================================================
# 11. Load saved checkpoint
# =================================================

print(
    "\nLoading checkpoint:"
)

print(
    CHECKPOINT_PATH
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


# =================================================
# 12. Restore model weights
# =================================================

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# =================================================
# 13. Move model to GPU
# =================================================

model = model.to(device)

model.eval()


# =================================================
# 14. Display checkpoint information
# =================================================

print(
    "\nCheckpoint loaded successfully."
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)


if "validation_results" in checkpoint:

    validation_results = (
        checkpoint[
            "validation_results"
        ]
    )

    print(
        "\nSaved validation results:"
    )

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

    print(
        "\nValidation confusion matrix:"
    )

    print(
        validation_results[
            "confusion_matrix"
        ]
    )


# =================================================
# 15. Verify parameter count
# =================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nModel parameters:",
    f"{total_params:,}"
)


# =================================================
# 16. Final held-out test evaluation
# =================================================

print(
    "\nEvaluating held-out test set..."
)

test_results = evaluate_model(
    model,
    test_loader,
    device
)


# =================================================
# 17. Display final test results
# =================================================

print(
    "\n"
    "========== FINAL TEST RESULTS =========="
)

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

print(
    "\nConfusion matrix:"
)

print(
    test_results[
        "confusion_matrix"
    ]
)


# =================================================
# 18. Summary
# =================================================

print(
    "\n"
    "========== BASELINE SUMMARY =========="
)

print(
    "Selected epoch:",
    checkpoint["epoch"]
)

if "validation_results" in checkpoint:

    print(
        "Best validation F1:",
        f"{checkpoint['validation_results']['f1']:.4f}"
    )

print(
    "Final test F1:",
    f"{test_results['f1']:.4f}"
)

print(
    "Final test accuracy:",
    f"{test_results['accuracy']:.4f}"
)
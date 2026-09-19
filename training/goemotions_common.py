"""Shared, explicit Study 2 data and training loop."""
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from data.goemotions import infer_schema, load_goemotions, prepare_study2_splits
from training.evaluate_multiclass import evaluate_multiclass

SEED = 42
NUM_CLASSES = 7
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 64


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def make_loaders(tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE):
    dataset = load_goemotions()
    text_field, label_field, label_names = infer_schema(dataset)
    train, validation, test = prepare_study2_splits(dataset, label_field)
    print(
        "Study 2 splits | "
        f"train={len(train):,} validation={len(validation):,} test={len(test):,} | "
        "published test preserved; validation is seed-42 stratified 10% of train"
    )

    def tokenize(batch):
        return tokenizer(batch[text_field], truncation=True, max_length=max_length)

    loaders = []
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    for split, shuffle in ((train, True), (validation, False), (test, False)):
        encoded = split.map(tokenize, batched=True)
        removable = [column for column in encoded.column_names if column not in {label_field, "input_ids", "attention_mask", "token_type_ids"}]
        encoded = encoded.remove_columns(removable)
        if label_field != "labels":
            encoded = encoded.rename_column(label_field, "labels")
        loaders.append(DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, collate_fn=collator))
    return (*loaders, label_names)


def print_results(title, results, label_names):
    print(f"\n{title}")
    print(json.dumps({key: value for key, value in results.items() if not key.startswith("per_class")}, indent=2))
    for index, name in enumerate(label_names):
        print(f"{name}: precision={results['per_class_precision'][index]:.4f}, recall={results['per_class_recall'][index]:.4f}, f1={results['per_class_f1'][index]:.4f}")


def train(model, tokenizer, checkpoint_path, model_name, max_length=MAX_LENGTH):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader, label_names = make_loaders(tokenizer, max_length)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_f1 = -1.0
    started = time.perf_counter()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, EPOCHS + 1):
        model.train(); loss_sum = 0.0; seen = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(**inputs), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * labels.size(0); seen += labels.size(0)
        results = evaluate_multiclass(model, validation_loader, device, criterion, NUM_CLASSES)
        print(f"Epoch {epoch} training loss: {loss_sum / seen:.4f}")
        print_results("Validation", results, label_names)
        if results["macro_f1"] > best_f1:
            best_f1 = results["macro_f1"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "validation_results": results, "label_names": label_names,
                "model_name": model_name, "seed": SEED, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "max_length": max_length,
            }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = evaluate_multiclass(model, test_loader, device, criterion, NUM_CLASSES)
    print_results("Held-out test", test_results, label_names)
    result_path = f"{checkpoint_path}.results.json"
    result_record = {
        "model_name": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": checkpoint["epoch"],
        "configuration": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_length": max_length,
            "checkpoint_metric": "validation macro_f1",
        },
        "label_names": label_names,
        "validation": checkpoint["validation_results"],
        "test": test_results,
    }
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result_record, result_file, indent=2)
    print("Machine-readable results:", result_path)
    print("Best epoch:", checkpoint["epoch"])
    print("Training and evaluation seconds:", time.perf_counter() - started)
    if use_amp:
        print("Peak allocated GPU bytes:", torch.cuda.max_memory_allocated())

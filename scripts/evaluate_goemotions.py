import torch
import torch.nn as nn
from training.goemotions_common import make_loaders, print_results
from training.evaluate_multiclass import evaluate_multiclass


def evaluate_checkpoint(model, tokenizer, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _, _, test_loader, label_names = make_loaders(tokenizer, checkpoint["max_length"], checkpoint["batch_size"])
    if label_names != checkpoint["label_names"]:
        raise ValueError("Dataset label order differs from the saved checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device)
    results = evaluate_multiclass(model, test_loader, device, nn.CrossEntropyLoss(), len(label_names))
    print_results("Held-out test", results, label_names)

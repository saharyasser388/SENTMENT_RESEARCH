import torch
from models.gpt2 import GPT_CONFIG_124M
from models.gpt2_classifier import GPT2Classifier

model = GPT2Classifier(GPT_CONFIG_124M, num_classes=7)
input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 8))
attention_mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])
logits = model(input_ids, attention_mask)
assert logits.shape == (2, 7)
assert model.classifier.out_features == 7
print("Output shape:", tuple(logits.shape))
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

from transformers import AutoTokenizer
from models.bert_classifier import BertClassifier

name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
batch = tokenizer(["I am delighted.", "This is frightening."], padding=True, return_tensors="pt")
model = BertClassifier(name, num_classes=7)
logits = model(**batch)
assert logits.shape == (2, 7)
assert batch["attention_mask"].shape == batch["input_ids"].shape
print("Output shape:", tuple(logits.shape))
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

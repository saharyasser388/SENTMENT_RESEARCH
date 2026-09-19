from datasets import load_dataset
from collections import Counter

dataset = load_dataset("nyu-mll/glue", "sst2")

print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])

print("\nDataset sizes:")
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))

print("\nLabel distribution:")
print("Train:", Counter(dataset["train"]["label"]))
print("Validation:", Counter(dataset["validation"]["label"]))
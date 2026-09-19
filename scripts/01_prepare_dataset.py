from datasets import load_dataset
from collections import Counter

# -------------------------------------------------
# 1. Load SST-2
# -------------------------------------------------

dataset = load_dataset("nyu-mll/glue", "sst2")

print("Original dataset:")
print(dataset)


# -------------------------------------------------
# 2. Create our own labeled test set
# -------------------------------------------------

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="label"
)

train_dataset = split["train"]
test_dataset = split["test"]

# Keep the official validation set
validation_dataset = dataset["validation"]


# -------------------------------------------------
# 3. Print final split sizes
# -------------------------------------------------

print("\nFinal dataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))


# -------------------------------------------------
# 4. Check label distributions
# -------------------------------------------------

print("\nTrain label distribution:")
print(Counter(train_dataset["label"]))

print("\nValidation label distribution:")
print(Counter(validation_dataset["label"]))

print("\nTest label distribution:")
print(Counter(test_dataset["label"]))


# -------------------------------------------------
# 5. Inspect one example
# -------------------------------------------------

print("\nExample training sample:")
print(train_dataset[0])
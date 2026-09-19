from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np


# -------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# -------------------------------------------------
# 2. Load SST-2
# -------------------------------------------------

dataset = load_dataset("nyu-mll/glue", "sst2")


# -------------------------------------------------
# 3. Recreate the same train/test split
# -------------------------------------------------

split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="label"
)

train_dataset = split["train"]


# -------------------------------------------------
# 4. Tokenize each sentence only to measure length
# -------------------------------------------------

lengths = []

for sentence in train_dataset["sentence"]:
    token_ids = tokenizer.encode(sentence)
    lengths.append(len(token_ids))


# -------------------------------------------------
# 5. Analyze sequence lengths
# -------------------------------------------------

lengths = np.array(lengths)

print("Number of training samples:", len(lengths))
print("Minimum length:", lengths.min())
print("Maximum length:", lengths.max())
print("Mean length:", round(lengths.mean(), 2))
print("Median length:", np.median(lengths))

print("\nPercentiles:")
print("90th percentile:", np.percentile(lengths, 90))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))

print("\nSamples longer than:")
for threshold in [32, 64, 128, 256]:
    count = np.sum(lengths > threshold)
    percentage = count / len(lengths) * 100

    print(
        f"{threshold} tokens: "
        f"{count} samples "
        f"({percentage:.2f}%)"
    )
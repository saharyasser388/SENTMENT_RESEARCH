import numpy as np
from data.goemotions import infer_schema, load_goemotions


TOKENIZATION_BATCH_SIZE = 1_000


def analyze(tokenizer, batch_size=TOKENIZATION_BATCH_SIZE):
    dataset = load_goemotions()
    text_field, _, _ = infer_schema(dataset)
    lengths = []
    for split in dataset.values():
        # A datasets.Column is iterable but is not a list[str], so recent
        # Transformers releases reject it as tokenizer input. Dataset.iter()
        # also avoids materializing an entire split in memory at once.
        for batch in split.iter(batch_size=batch_size):
            texts = list(batch[text_field])
            encoded = tokenizer(texts, truncation=False)
            lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    if not lengths:
        raise ValueError("Cannot analyze token lengths for an empty dataset")
    values = np.asarray(lengths)
    print("Examples:", len(values))
    print("Minimum:", values.min(), "Maximum:", values.max())
    print("Mean:", values.mean(), "Median:", np.median(values))
    for percentile in (90, 95, 99):
        print(f"P{percentile}:", np.percentile(values, percentile))
    for threshold in (32, 64, 128, 256):
        count = int((values > threshold).sum())
        print(f"> {threshold}: {count} ({100 * count / len(values):.2f}%)")

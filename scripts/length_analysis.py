import numpy as np
from data.goemotions import infer_schema, load_goemotions


def analyze(tokenizer):
    dataset = load_goemotions()
    text_field, _, _ = infer_schema(dataset)
    lengths = []
    for split in dataset.values():
        lengths.extend(len(ids) for ids in tokenizer(split[text_field], truncation=False)["input_ids"])
    values = np.asarray(lengths)
    print("Examples:", len(values))
    print("Minimum:", values.min(), "Maximum:", values.max())
    print("Mean:", values.mean(), "Median:", np.median(values))
    for percentile in (90, 95, 99):
        print(f"P{percentile}:", np.percentile(values, percentile))
    for threshold in (32, 64, 128, 256):
        count = int((values > threshold).sum())
        print(f"> {threshold}: {count} ({100 * count / len(values):.2f}%)")

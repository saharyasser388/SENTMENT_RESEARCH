"""Shared dataset constants and schema helpers for Study 2."""
from datasets import ClassLabel, DatasetDict, load_dataset

DATASET_NAME = "Jsevisal/go_emotions_ekman_unilabel"
SEED = 42
EXPECTED_LABEL_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def load_goemotions():
    return load_dataset(DATASET_NAME)


def infer_schema(dataset: DatasetDict):
    first_split = next(iter(dataset.values()))
    features = first_split.features
    text_fields = [name for name, feature in features.items() if getattr(feature, "dtype", None) == "string"]
    label_fields = [name for name, feature in features.items() if isinstance(feature, ClassLabel)]
    if len(text_fields) != 1 or len(label_fields) != 1:
        raise ValueError(f"Expected one text and one ClassLabel field; got {features}")
    label_feature = features[label_fields[0]]
    if len(label_feature.names) != 7:
        raise ValueError(f"Expected 7 classes, got {label_feature.names}")
    return text_fields[0], label_fields[0], list(label_feature.names)


def require_standard_splits(dataset: DatasetDict):
    required = {"train", "validation", "test"}
    missing = required.difference(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset is missing {sorted(missing)}. Inspect it before defining a shared "
            "seed-42 stratified split; do not train with model-specific splits."
        )
    return dataset["train"], dataset["validation"], dataset["test"]

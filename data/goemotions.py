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


def prepare_study2_splits(dataset: DatasetDict, label_field, validation_size=0.1):
    """Return the one canonical split policy used by both Study 2 models.

    The published dataset has train and test splits but no validation split.
    Preserve its held-out test set and stratify 10% of the published training
    split into validation data with seed 42. If a future dataset revision adds
    validation data, use all three published splits without resplitting them.
    """
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(
            "Study 2 requires published train and test splits; got "
            f"{sorted(dataset.keys())}."
        )
    if "validation" in dataset:
        return dataset["train"], dataset["validation"], dataset["test"]

    split = dataset["train"].train_test_split(
        test_size=validation_size,
        seed=SEED,
        stratify_by_column=label_field,
    )
    return split["train"], split["test"], dataset["test"]


def require_standard_splits(dataset: DatasetDict):
    """Compatibility wrapper for checkouts using the original helper name.

    The first Study 2 revision required a published validation split. Keeping
    this name prevents a partially updated checkout from silently retaining
    that obsolete behavior; it now delegates to the canonical split policy.
    New code should call :func:`prepare_study2_splits` with its known label
    field instead.
    """
    _, label_field, _ = infer_schema(dataset)
    return prepare_study2_splits(dataset, label_field)

from collections import Counter
from datasets import ClassLabel
from data.goemotions import infer_schema, load_goemotions, prepare_study2_splits


def main():
    dataset = load_goemotions()
    print("Available splits:", list(dataset.keys()))
    print("Features:", dataset[next(iter(dataset))].features)
    text_field, label_field, label_names = infer_schema(dataset)
    print("Text field:", text_field)
    print("Label field:", label_field)
    print("Label names/order:", label_names)
    single_label = isinstance(dataset[next(iter(dataset))].features[label_field], ClassLabel)
    print("Single-label:", single_label)
    for split_name, split in dataset.items():
        print(f"\n{split_name}: {len(split):,} rows")
        print("Examples:", split.select(range(min(3, len(split))))[:])
        counts = Counter(split[label_field])
        print("Class distribution:", {label_names[i]: counts[i] for i in range(len(label_names))})
    custom_split_needed = "validation" not in dataset
    print("Custom validation split needed:", custom_split_needed)
    train, validation, test = prepare_study2_splits(dataset, label_field)
    print("Final shared split sizes:")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))
    if custom_split_needed:
        print("Split policy: stratified 90/10 split of published train, seed 42")
        print("The published test split remains unchanged.")


if __name__ == "__main__":
    main()

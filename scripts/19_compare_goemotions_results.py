"""Print a side-by-side comparison from two completed Study 2 runs."""
import json

GPT2_RESULTS = "checkpoints/study2_gpt2_goemotions_best.pt.results.json"
BERT_RESULTS = "checkpoints/study2_bert_goemotions_best.pt.results.json"


def load_results(path):
    try:
        with open(path, encoding="utf-8") as result_file:
            return json.load(result_file)
    except FileNotFoundError as error:
        raise SystemExit(
            f"Missing {path}. Complete both training runs before generating the comparison."
        ) from error


def main():
    gpt2 = load_results(GPT2_RESULTS)
    bert = load_results(BERT_RESULTS)
    if gpt2["label_names"] != bert["label_names"]:
        raise SystemExit("The runs used different class mappings and cannot be compared.")

    print("| Metric | GPT-2 | BERT-base-uncased |")
    print("| --- | ---: | ---: |")
    rows = (
        ("Parameters", "parameter_count", None),
        ("Best epoch", "best_epoch", None),
        ("Validation accuracy", "accuracy", "validation"),
        ("Validation Macro-F1", "macro_f1", "validation"),
        ("Test accuracy", "accuracy", "test"),
        ("Test macro precision", "macro_precision", "test"),
        ("Test macro recall", "macro_recall", "test"),
        ("Test Macro-F1", "macro_f1", "test"),
        ("Test weighted F1", "weighted_f1", "test"),
    )
    for title, key, section in rows:
        left = gpt2[key] if section is None else gpt2[section][key]
        right = bert[key] if section is None else bert[section][key]
        if isinstance(left, float):
            left, right = f"{left:.4f}", f"{right:.4f}"
        print(f"| {title} | {left} | {right} |")

    print("\n| Emotion | GPT-2 F1 | BERT F1 |")
    print("| --- | ---: | ---: |")
    for index, emotion in enumerate(gpt2["label_names"]):
        print(
            f"| {emotion} | {gpt2['test']['per_class_f1'][index]:.4f} | "
            f"{bert['test']['per_class_f1'][index]:.4f} |"
        )

    print("\nGPT-2 confusion matrix:", gpt2["test"]["confusion_matrix"])
    print("BERT confusion matrix:", bert["test"]["confusion_matrix"])


if __name__ == "__main__":
    main()

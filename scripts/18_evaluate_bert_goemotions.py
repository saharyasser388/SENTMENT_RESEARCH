from transformers import AutoTokenizer
from models.bert_classifier import BertClassifier
from scripts.evaluate_goemotions import evaluate_checkpoint

if __name__ == "__main__":
    name = "google-bert/bert-base-uncased"
    evaluate_checkpoint(BertClassifier(name, 7), AutoTokenizer.from_pretrained(name), "checkpoints/study2_bert_goemotions_best.pt")

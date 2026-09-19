from transformers import AutoTokenizer
from scripts.length_analysis import analyze

if __name__ == "__main__":
    analyze(AutoTokenizer.from_pretrained("google-bert/bert-base-uncased"))

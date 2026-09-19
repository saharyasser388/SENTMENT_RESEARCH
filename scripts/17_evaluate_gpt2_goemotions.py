from transformers import GPT2TokenizerFast
from models.gpt2 import GPT_CONFIG_124M
from models.gpt2_classifier import GPT2Classifier
from scripts.evaluate_goemotions import evaluate_checkpoint

if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tokenizer.pad_token = tokenizer.eos_token
    evaluate_checkpoint(GPT2Classifier(GPT_CONFIG_124M, 7), tokenizer, "checkpoints/study2_gpt2_goemotions_best.pt")

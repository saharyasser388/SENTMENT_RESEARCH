from transformers import GPT2TokenizerFast
from scripts.length_analysis import analyze

if __name__ == "__main__":
    analyze(GPT2TokenizerFast.from_pretrained("openai-community/gpt2"))

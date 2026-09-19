from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from models.gpt2 import GPT_CONFIG_124M, load_hf_weights_into_gpt
from models.gpt2_classifier import GPT2Classifier
from training.goemotions_common import train

MODEL_NAME = "openai-community/gpt2"
CHECKPOINT_PATH = "checkpoints/study2_gpt2_goemotions_best.pt"


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Classifier(GPT_CONFIG_124M, num_classes=7)
    pretrained = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    load_hf_weights_into_gpt(model.gpt, pretrained)
    del pretrained
    train(model, tokenizer, CHECKPOINT_PATH, MODEL_NAME)


if __name__ == "__main__":
    main()

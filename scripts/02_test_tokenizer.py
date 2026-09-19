from transformers import GPT2TokenizerFast

# -------------------------------------------------
# 1. Load the pretrained GPT-2 tokenizer
# -------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# -------------------------------------------------
# 2. Inspect basic tokenizer information
# -------------------------------------------------

print("Vocabulary size:", tokenizer.vocab_size)
print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token)


# -------------------------------------------------
# 3. Test tokenization on one sentence
# -------------------------------------------------

sentence = "I really enjoyed this movie."

encoded = tokenizer(sentence)

print("\nOriginal sentence:")
print(sentence)

print("\nToken IDs:")
print(encoded["input_ids"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

print("\nDecoded back to text:")
print(tokenizer.decode(encoded["input_ids"]))


# -------------------------------------------------
# 4. Inspect another sentence
# -------------------------------------------------

sentence_2 = "This movie was absolutely terrible!"

encoded_2 = tokenizer(sentence_2)

print("\nSecond sentence:")
print(sentence_2)

print("\nToken IDs:")
print(encoded_2["input_ids"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded_2["input_ids"]))

print("\nDecoded:")
print(tokenizer.decode(encoded_2["input_ids"]))
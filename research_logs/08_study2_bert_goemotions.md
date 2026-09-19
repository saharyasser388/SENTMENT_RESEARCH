# Study 2: BERT GoEmotions

## Objective
Fine-tune BERT-base-uncased for the same seven-class task.

## Why this step is needed
This supplies the encoder-only comparator under the same split and evaluation policy.

## Configuration
Seed 42; 3 epochs; batch size 8; AdamW; learning rate 2e-5; weight decay 0.01; cross-entropy; CUDA mixed precision when available; validation Macro-F1 selection. Provisional maximum length: 64, pending measured length analysis.

## Architecture
Hugging Face `google-bert/bert-base-uncased`; first-token (`[CLS]`) final hidden state; `Linear(768, 7)`. All parameters are trainable.

## Dataset
Exactly the same published test split and seed-42 stratified 90/10 training/validation partition as GPT-2, produced by the shared split helper.

## What changed
Added the model, pretrained-load smoke test, training entry point, and held-out evaluation script.

## Results
Not run because remote Hugging Face artifacts were blocked. No metrics or checkpoint are claimed.

## Observations
The model-specific tokenizer is intentionally retained; all experimental policy settings otherwise match GPT-2.

## Problems encountered
The configured proxy returned HTTP 403 for Hugging Face access. When
`BertModel` is loaded from the upstream pretraining checkpoint, Transformers
may report the `cls.predictions.*` and `cls.seq_relationship.*` keys as
unexpected. Those keys belong to BERT's masked-language-model and
next-sentence-prediction pretraining heads; this study intentionally loads the
backbone and supplies its own seven-class head, so that load report is not a
training error.

## Interpretation
No comparison is possible until both complete runs exist.

## Next step
Run `python -m training.train_bert_goemotions` after schema/length confirmation and the GPT-2 baseline.

# Study 2: GPT-2 GoEmotions

## Objective
Fine-tune the validated custom GPT-2 Small backbone for seven-class emotion classification.

## Why this step is needed
This supplies the decoder-only side of the controlled comparison.

## Configuration
Seed 42; 3 epochs; batch size 8; AdamW; learning rate 2e-5; weight decay 0.01; cross-entropy; CUDA mixed precision when available; validation Macro-F1 checkpoint selection. The provisional maximum length is 64 and must be confirmed by scripts 13 and 14 before a research run.

## Architecture
Custom pretrained GPT-2 Small, last non-padding token representation, and `Linear(768, 7)`. All parameters are trainable.

## Dataset
The published test split is retained. The published training split is divided identically for both models into stratified 90% training and 10% validation partitions with seed 42 through `data/goemotions.py`.

## What changed
Added a seven-class smoke test, training entry point, checkpoint metadata, and held-out checkpoint evaluator. The existing generic classifier remains backward compatible with Study 1's two-class default.

## Results
Not run: dataset/model downloads were blocked by the environment proxy. No checkpoint or metrics are claimed.

## Observations
The local randomly initialized classifier smoke test passed with output shape `[2, 7]`.

## Problems encountered
Remote artifacts were unavailable, and this environment has no verified full training run.

## Interpretation
Implementation readiness is not an empirical model result.

## Next step
Confirm length 64 from measured distributions, then run `python -m training.train_gpt2_goemotions` on CUDA and paste emitted metrics here.

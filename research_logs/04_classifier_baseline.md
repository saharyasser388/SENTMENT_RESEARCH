# GPT-2 Sentiment Classifier Baseline

## Objective

Adapt the validated pretrained GPT-2 Small implementation from next-token prediction to binary sentiment classification.

## Classification Architecture

The language-model vocabulary head is not used for sentiment classification.

Instead, the model uses:

Pretrained GPT-2
→ Final hidden states
→ Last real token representation
→ Linear classification head
→ Two sentiment logits

The classification labels are:

* `0` = Negative
* `1` = Positive

## Sentence Representation

GPT-2 produces a 768-dimensional contextual representation for every input token.

Because GPT-2 uses causal attention, the final real token can attend to all previous real tokens in the sequence.

Therefore, the hidden representation of the final non-padding token is used as the sentence representation for the baseline classifier.

## Classification Head

The classification head is:

`Linear(768, 2)`

It contains:

* 1,536 weights
* 2 bias parameters

Total newly introduced parameters:

`1,538`

Complete model parameters:

`124,441,346`

## Architecture Verification

The classifier was tested successfully using example sentences.

Input shape:

`[2, 12]`

Classifier output shape:

`[2, 2]`

This verifies that one pair of binary sentiment logits is produced for each sentence.

The initial sentiment predictions were not interpreted because the classification head was randomly initialized.

## Training Data Pipeline

The SST-2 training pipeline uses:

Raw sentence
→ GPT-2 tokenizer
→ truncation at 64 tokens
→ dynamic padding
→ attention mask
→ PyTorch DataLoader

Configuration:

* Training samples: 60,614
* Validation samples: 872
* Test samples: 6,735
* Batch size: 8
* Training batches: 7,577
* Validation batches: 109
* Test batches: 842

## Classifier-Only Training Sanity Check

Before full fine-tuning, the GPT-2 backbone was frozen and only the classification head was trained.

Trainable parameters:

`1,538`

Optimizer:

`AdamW`

Learning rate:

`1e-3`

Loss:

`CrossEntropyLoss`

Training duration:

`100 optimization steps`

Observed average losses:

| Step | Average Loss |
| ---: | -----------: |
|   10 |       0.7563 |
|   20 |       0.7227 |
|   30 |       0.7238 |
|   40 |       0.7040 |
|   50 |       0.6928 |
|   60 |       0.6844 |
|   70 |       0.6653 |
|   80 |       0.6379 |
|   90 |       0.6511 |
|  100 |       0.6581 |

## Interpretation

The loss did not decrease monotonically, which is normal for mini-batch stochastic optimization.

However, the overall loss decreased from approximately `0.76` during the first reported interval to approximately `0.66` during the final interval.

This sanity check was not intended to measure final sentiment-classification quality.

Its purpose was to verify that:

* data loading works,
* tokenization and padding work,
* labels are correctly supplied,
* model forward propagation works,
* cross-entropy loss is calculated,
* gradients are produced,
* optimizer updates occur,
* the classifier parameters can learn.

The sanity check therefore confirms that the training pipeline is functional.

## Important Note

This classifier-only result is not considered the primary GPT-2 baseline experiment.

The actual baseline will fine-tune the pretrained GPT-2 backbone together with the classification head.

Evaluation Pipeline

A reusable evaluation function was implemented using:

- Cross-entropy validation loss
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For binary metrics, label 1 is treated as the positive class.

The evaluation implementation is shared across model variants to maintain consistent measurement between the GPT-2 baseline, GPT-2 + MoE, GPT-2 + Mamba, and combined architectures.

## Next Step

Perform full GPT-2 sentiment fine-tuning and evaluate the model on the validation and held-out test sets.

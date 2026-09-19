# GPT-2 Baseline Fine-Tuning

## Objective

Fine-tune the complete pretrained GPT-2 Small model for binary sentiment classification on SST-2 and establish the primary baseline for later Mixture-of-Experts and Mamba experiments.

## Model Architecture

The baseline architecture is:

Pretrained GPT-2 Small
→ Final hidden states
→ Last real-token representation
→ Linear classification head
→ Negative / Positive

The pretrained GPT-2 backbone and the classification head were fine-tuned jointly.

## Dataset

SST-2 binary sentiment classification.

Labels:

* `0` = Negative
* `1` = Positive

Dataset sizes:

* Training: 60,614 samples
* Validation: 872 samples
* Held-out test: 6,735 samples

The held-out test set was created using a stratified 10% split from the original SST-2 training set using random seed 42.

## Training Configuration

* Model: GPT-2 Small
* Total parameters: 124,441,346
* Trainable parameters: 124,441,346
* Maximum sequence length: 64
* Dynamic padding: Yes
* Batch size: 8
* Epochs: 3
* Optimizer: AdamW
* Learning rate: `2e-5`
* Weight decay: `0.01`
* Loss function: CrossEntropyLoss
* Random seed: 42
* Mixed precision: Enabled
* Checkpoint-selection metric: Validation F1

## Training Results

### Epoch 1

Validation loss:

`0.2264`

Validation accuracy:

`0.9140`

Validation precision:

`0.9321`

Validation recall:

`0.8964`

Validation F1:

`0.9139`

Confusion matrix:

`[[399, 29], [46, 398]]`

### Epoch 2

Validation loss:

`0.2587`

Validation accuracy:

`0.9209`

Validation precision:

`0.9085`

Validation recall:

`0.9392`

Validation F1:

`0.9236`

Confusion matrix:

`[[386, 42], [27, 417]]`

### Epoch 3

Validation loss:

`0.2591`

Validation accuracy:

`0.9014`

Validation precision:

`0.9032`

Validation recall:

`0.9032`

Validation F1:

`0.9032`

Confusion matrix:

`[[385, 43], [43, 401]]`

## Best Checkpoint

The best checkpoint was selected using validation F1.

Selected epoch:

`Epoch 2`

Best validation F1:

`0.9236`

Although training loss continued to decrease during Epoch 3, validation performance decreased.

This suggests that the model had begun to overfit the training data after Epoch 2.

Therefore, the Epoch 2 checkpoint was selected for final testing.

## Final Held-Out Test Results

Test loss:

`0.1569`

Accuracy:

`0.9467`

Precision:

`0.9553`

Recall:

`0.9489`

F1-score:

`0.9521`

Confusion matrix:

`[[2811, 167], [192, 3565]]`

The confusion matrix corresponds to:

* True negatives: 2,811
* False positives: 167
* False negatives: 192
* True positives: 3,565

Correct predictions:

`6,376`

Incorrect predictions:

`359`

Total test samples:

`6,735`

## Interpretation

The pretrained GPT-2 Small model was successfully adapted to binary sentiment classification.

Fine-tuning substantially improved the model beyond the randomly initialized classification head and established a strong baseline for subsequent architectural experiments.

Validation F1 peaked after two epochs and declined during the third epoch despite continued reductions in training loss. This indicates that additional training did not improve generalization and provides evidence of beginning overfitting.

The final held-out test F1 of `0.9521` and accuracy of `0.9467` will serve as the primary reference values for later experiments.

The held-out test set achieved higher scores than the official validation set. This is possible because the two evaluation sets originate from different sampling procedures and may differ in difficulty. The test result should therefore be interpreted as performance on the specific stratified held-out SST-2 subset used in this study.

## Baseline Reference

Future architectures will be compared against:

| Metric              | GPT-2 Baseline |
| ------------------- | -------------: |
| Parameters          |    124,441,346 |
| Validation Accuracy |         0.9209 |
| Validation F1       |         0.9236 |
| Test Accuracy       |         0.9467 |
| Test Precision      |         0.9553 |
| Test Recall         |         0.9489 |
| Test F1             |         0.9521 |

## Next Step

Introduce a Mixture-of-Experts component into the GPT-2 architecture and evaluate whether conditional expert computation changes sentiment-classification performance or computational characteristics.

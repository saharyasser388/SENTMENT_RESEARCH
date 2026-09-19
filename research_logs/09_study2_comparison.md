# Study 2: GPT-2 vs BERT Comparison

> **Status: results not yet available.** The result cells below are not hidden
> elsewhere in the repository: neither training run completed because this
> environment could not download the dataset or pretrained models. After both
> runs complete, their machine-readable results will be written beside the
> checkpoints as `*.results.json`. Run
> `python -m scripts.19_compare_goemotions_results` to print the comparison
> table and per-class scores. The command fails loudly if either run is absent.

## Objective
Compare decoder-only GPT-2 Small and encoder-only BERT-base-uncased on one seven-class GoEmotions-Ekman task.

## Why this step is needed
It addresses the Study 2 research question while controlling the split, seed, epochs, optimizer, learning rate, loss, checkpoint metric, and evaluation implementation.

## Configuration
Both planned runs use seed 42, 3 epochs, batch size 8, AdamW at 2e-5 with weight decay 0.01, dynamic padding, cross-entropy, CUDA AMP, and validation Macro-F1 selection.

## Architecture
GPT-2 uses its last real token; BERT uses its `[CLS]` hidden state. Both attach a 768-to-7 linear classifier.

## Dataset
Both entry points call the same loader. They preserve the published test set and use the same seed-42 stratified 90/10 partition of the published training split for training and validation. Exact resulting sizes and class mapping remain pending a recorded inspection run.

## What changed
A common multiclass evaluator reports accuracy, macro precision/recall/F1, weighted F1, ordered per-class scores, and a 7-by-7 confusion matrix.

## Results

There are currently no measured comparison results. The following table is a
report template, not a completed experiment.
| Metric | GPT-2 | BERT-base-uncased |
| --- | ---: | ---: |
| Parameters | Not measured | Not measured |
| Best Epoch | Not run | Not run |
| Validation Accuracy | Not run | Not run |
| Validation Macro-F1 | Not run | Not run |
| Test Accuracy | Not run | Not run |
| Test Macro Precision | Not run | Not run |
| Test Macro Recall | Not run | Not run |
| Test Macro-F1 | Not run | Not run |
| Test Weighted F1 | Not run | Not run |

| Emotion | GPT-2 F1 | BERT F1 |
| --- | ---: | ---: |
| Anger | Not run | Not run |
| Disgust | Not run | Not run |
| Fear | Not run | Not run |
| Joy | Not run | Not run |
| Sadness | Not run | Not run |
| Surprise | Not run | Not run |
| Neutral | Not run | Not run |

Confusion matrices, training behavior, elapsed time, inference time, and peak CUDA memory are not available. Training records elapsed training/evaluation time and peak allocated CUDA memory when applicable; dedicated inference timing was not added because no run could validate it.

## Observations
No architecture ranking can be made. A single future baseline run per model will support a scoped comparison, not a universal superiority claim.

## Problems encountered
Hugging Face access failed with an HTTP 403 proxy response; consequently the acceptance criteria requiring training and measured results remain outstanding.

## Interpretation
The reproducible implementation is present, while empirical completion is explicitly deferred rather than fabricated.

## Next step
For Study 3, after completing these baselines, repeat runs across multiple fixed seeds or test a parameter-efficient adaptation while preserving this split and evaluator.

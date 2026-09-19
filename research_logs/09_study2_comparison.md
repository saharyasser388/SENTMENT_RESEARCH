# Study 2: GPT-2 vs BERT Comparison

## Objective
Compare decoder-only GPT-2 Small and encoder-only BERT-base-uncased on one seven-class GoEmotions-Ekman task.

## Why this step is needed
It addresses the Study 2 research question while controlling the split, seed, epochs, optimizer, learning rate, loss, checkpoint metric, and evaluation implementation.

## Configuration
Both planned runs use seed 42, 3 epochs, batch size 8, AdamW at 2e-5 with weight decay 0.01, dynamic padding, cross-entropy, CUDA AMP, and validation Macro-F1 selection.

## Architecture
GPT-2 uses its last real token; BERT uses its `[CLS]` hidden state. Both attach a 768-to-7 linear classifier.

## Dataset
Both entry points call the same loader and require upstream train/validation/test splits. Exact sizes and class mapping remain pending successful inspection.

## What changed
A common multiclass evaluator reports accuracy, macro precision/recall/F1, weighted F1, ordered per-class scores, and a 7-by-7 confusion matrix.

## Results
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

# Study 2: GoEmotions-Ekman Dataset

## Objective
Inspect `Jsevisal/go_emotions_ekman_unilabel`, confirm its schema and class order, and fix one split policy for both models.

## Why this step is needed
A shared, single-label, seven-class dataset is necessary for a fair architecture comparison.

## Configuration
Seed: 42. Inspection command: `python -m scripts.12_inspect_goemotions`.

## Architecture
Not applicable.

## Dataset
The requested Hugging Face dataset is used directly. The code discovers the sole string field and `ClassLabel` field rather than assuming their names. The published dataset has `train` and `test` splits but no `validation` split. Study 2 preserves the published test set and deterministically creates validation data with a stratified 90/10 split of the published training data using seed 42. Both models call this same helper.

## What changed
Added shared schema validation, deterministic split preparation, dataset inspection, and separate GPT-2/BERT length-analysis scripts.

## Results
Not measured. The execution environment returned HTTP 403 from its configured proxy while contacting Hugging Face, so split sizes, field names, class order, distributions, and token-length statistics could not be truthfully recorded.

## Observations
The inspection script prints all requested evidence. Training cannot proceed until its output confirms the remote schema. The expected semantic classes are anger, disgust, fear, joy, sadness, surprise, and neutral, but their encoded order is deliberately not asserted without measurement.

## Problems encountered
Hugging Face dataset access was blocked by the environment proxy.

## Interpretation
No dataset or experimental result is claimed. This is intentionally an incomplete measurement record rather than invented data.

## Next step
Run scripts 12–14 with Hugging Face access, record their output here, and set the shared maximum length from both measured distributions before training.

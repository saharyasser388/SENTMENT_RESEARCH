# Repository Guidelines

## Project Structure & Module Organization

This repository studies GPT-2 sentiment classification on SST-2 using PyTorch and Hugging Face.

- `models/`: custom GPT-2 implementation, pretrained-weight loading, and classification head.
- `training/`: baseline fine-tuning and reusable evaluation logic.
- `scripts/`: numbered experiments for dataset preparation, tokenization, model checks, and evaluation.
- `data/`: dataset-loading code; `notebooks/`: exploratory work.
- `research_logs/`: numbered Markdown records of experiment setup and results.
- `checkpoints/`: generated model weights, excluded by `.gitignore`.

## Build, Test, and Development Commands

Run commands from the repository root. Set up a virtual environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

There is no separate build step. Common entry points are:

- `python -m scripts.01_prepare_dataset`: inspect SST-2 splits and label distributions.
- `python -m scripts.08_prepare_dataloaders`: verify tokenization, padding, and batch shapes.
- `python -m scripts.09_train_classifier_sanity`: run a short classifier-head training check.
- `python -m training.train_baseline`: fine-tune the baseline model.
- `python -m scripts.11_evaluate_saved_baseline`: evaluate the saved checkpoint; confirm its configured path exists first.

Initial runs may download datasets, tokenizer files, and pretrained weights.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Follow existing multiline formatting. Keep reusable logic in `models/` and `training/`. Name experiment scripts `NN_descriptive_name.py` and research logs `NN_descriptive_name.md`. No formatter or linter is configured.

## Testing Guidelines

Validation uses executable sanity scripts; no dedicated test framework or coverage threshold is configured. Run checks relevant to the change, such as `python -m scripts.07_test_classifier` and `python -m scripts.10_test_evaluator`. Inspect shapes, losses, and metrics rather than relying solely on process completion.

Preserve seed 42 and the stratified 10% held-out training split for comparable experiments. Select checkpoints using validation F1; reserve held-out test results for final evaluation. Record configuration and metrics in `research_logs/`.

## Commit & Pull Request Guidelines

Existing commits use short informal subjects, with no established prefix convention. Write concise, action-oriented subjects. Pull requests should explain the purpose, affected experiments, commands run, and metric changes. Link relevant issues and research logs. Keep generated checkpoints, caches, and local environment files out of commits.
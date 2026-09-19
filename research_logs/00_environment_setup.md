# Environment Setup

## Objective

Prepare and document the software and hardware environment used for the GPT-2 sentiment classification experiments.

## Why This Step Is Important

A documented environment improves reproducibility and makes it possible to recreate the experimental setup later.

## Environment

* Operating System: Windows
* Python Version: 3.12.10
* Virtual Environment: `.venv`
* pip Version: 26.2.1
* GPU: NVIDIA GeForce RTX 2060
* GPU Memory: 6 GB
* NVIDIA Driver: 616.64
* CUDA version reported by NVIDIA driver: 13.4
* PyTorch Version: 2.14.0+cu126
* PyTorch CUDA Runtime: 12.6
* CUDA Available in PyTorch: Yes

## Virtual Environment Location

`F:\sahar\gpt2_sentiment_research\.venv`

## Verification

The following checks were completed successfully:

* Python virtual environment is active.
* pip is installed inside the virtual environment.
* PyTorch imports successfully.
* PyTorch detects CUDA successfully.
* PyTorch detects the NVIDIA GeForce RTX 2060.
* GPU acceleration is available for model training.

## Issue Encountered

The initial PyTorch installation was CPU-only:

`2.14.0+cpu`

This resulted in:

`torch.cuda.is_available() = False`

PyTorch was reinstalled using the CUDA 12.6 build:

`2.14.0+cu126`

After reinstalling, CUDA detection succeeded.

## Result

The environment is ready for GPU-accelerated GPT-2 experiments.

## Next Step

Install the remaining NLP and evaluation libraries required for dataset preparation and sentiment classification.

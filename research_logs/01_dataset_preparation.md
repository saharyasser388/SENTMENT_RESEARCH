# Dataset Preparation

## Objective

Prepare and inspect the sentiment classification dataset used for the GPT-2 baseline.

## Dataset

Stanford Sentiment Treebank 2 (SST-2), accessed through the GLUE dataset collection.

Dataset repository:

`nyu-mll/glue`

Configuration:

`sst2`

## Task

Binary sentiment classification.

## Labels

* `0` = Negative
* `1` = Positive

## Why SST-2 Was Selected

SST-2 provides a relatively simple binary sentiment classification task and is suitable for establishing a clean GPT-2 baseline before introducing architectural modifications such as Mixture-of-Experts and Mamba.

Its relatively short sentences also make initial training and debugging easier.

## Dataset Structure

Each example contains:

* `sentence`: input text
* `label`: sentiment class
* `idx`: sample identifier

Example:

`{'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': 0}`

## Dataset Sizes

* Training samples: 67,349
* Validation samples: 872
* Official test samples: 1,821

## Class Distribution

### Training Set

* Positive: 37,569
* Negative: 29,780

### Validation Set

* Positive: 444
* Negative: 428

## Observation

The dataset is reasonably balanced. There is no severe class imbalance that requires special handling for the baseline experiment.

The validation set is particularly balanced between positive and negative samples.

## Test Set Consideration

The official SST-2 test split does not provide usable gold labels for standard local evaluation.

Therefore, a labeled test set will be created by holding out part of the original training data.

The official validation set will remain unchanged and will be used during model development.

## Issue Encountered

The initial dataset identifier:

`glue`

failed with the current Hugging Face Hub URI format.

The namespaced repository identifier was used instead:

`nyu-mll/glue`

Final loading command:

`load_dataset("nyu-mll/glue", "sst2")`

## Result

The SST-2 dataset was successfully loaded and inspected.

The dataset is suitable for the initial GPT-2 sentiment classification baseline.

## Next Step

Prepare the final train, validation, and test splits and then tokenize the sentences using the GPT-2 tokenizer.

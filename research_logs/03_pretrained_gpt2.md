# Pretrained GPT-2 Weight Transfer

## Objective

Load pretrained GPT-2 Small parameters into the custom PyTorch GPT-2 implementation and verify that the custom architecture reproduces the behavior of a reference pretrained GPT-2 model.

## Why This Step Is Important

The custom GPT-2 architecture was initially created with randomly initialized parameters.

Training GPT-2 from scratch is outside the scope of this research and would require substantially more data and computational resources.

Instead, pretrained GPT-2 parameters are transferred into the custom implementation so that the model begins with previously learned language representations.

The custom architecture is retained because later experiments require direct modification of internal components for Mixture-of-Experts and Mamba.

## Pretrained Model

Reference checkpoint:

`openai-community/gpt2`

Model size:

GPT-2 Small

Architecture:

* Vocabulary size: 50,257
* Embedding dimension: 768
* Transformer blocks: 12
* Attention heads: 12
* Maximum context length: 1,024
* Total parameters: 124,439,808

## Weight Transfer Strategy

Hugging Face's pretrained `GPT2LMHeadModel` was used as the source of pretrained parameters.

The following parameters were transferred:

* Token embeddings
* Position embeddings
* Query projection
* Key projection
* Value projection
* Attention output projection
* Feed-forward expansion layer
* Feed-forward contraction layer
* Transformer LayerNorm parameters
* Final LayerNorm parameters

The output vocabulary projection shares its weights with the token embedding matrix through weight tying.

## QKV Conversion

The reference GPT-2 implementation stores Query, Key, and Value projections in a combined QKV matrix.

The custom implementation stores these projections separately.

Therefore, the combined pretrained matrix was split into:

* Query weights
* Key weights
* Value weights

and assigned to the corresponding custom attention layers.

## Weight Orientation

The reference GPT-2 implementation uses Conv1D-style projection layers, whose stored matrix orientation differs from PyTorch `nn.Linear`.

The corresponding pretrained weight matrices were transposed before being copied into the custom model.

## Verification Method

The same tokenized input batch was passed through:

1. Hugging Face pretrained GPT-2
2. Custom GPT-2 containing transferred pretrained weights

The vocabulary logits were compared only at real, non-padding token positions.

## Verification Results

Reference model output shape:

`[2, 6, 50257]`

Custom model output shape:

`[2, 6, 50257]`

Maximum absolute difference:

`7.62939453125e-05`

Mean absolute difference:

`1.8125851056538522e-05`

Approximate equality test:

`True`

Weight tying verification:

`True`

Custom model parameter count:

`124,439,808`

## Interpretation

The very small numerical differences are consistent with minor floating-point variation between equivalent implementations.

The approximate equality test succeeded, providing evidence that the pretrained parameters were transferred correctly and that the custom GPT-2 implementation behaves consistently with the reference pretrained model.

## Result

The custom GPT-2 Small architecture is now validated and initialized with pretrained GPT-2 parameters.

It is ready to be adapted from next-token prediction to sentiment classification.

## Next Step

Replace the language-model output objective with a binary sentiment classification head and verify the new classifier architecture before beginning fine-tuning.

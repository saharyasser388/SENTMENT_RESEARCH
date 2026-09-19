import torch
import torch.nn as nn

from models.gpt2 import GPTModel


class GPT2Classifier(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super().__init__()

        self.gpt = GPTModel(cfg)

        self.classifier = nn.Linear(
            cfg["emb_dim"],
            num_classes
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # ---------------------------------------------
        # 1. Get contextual GPT-2 hidden states
        # ---------------------------------------------

        hidden_states = self.gpt.forward_features(
            input_ids,
            attention_mask=attention_mask
        )

        # Shape:
        # [batch_size, sequence_length, 768]


        # ---------------------------------------------
        # 2. Find last real token
        # ---------------------------------------------

        last_token_indices = (
            attention_mask.sum(dim=1) - 1
        )


        # ---------------------------------------------
        # 3. Select its representation
        # ---------------------------------------------

        batch_indices = torch.arange(
            input_ids.size(0),
            device=input_ids.device
        )

        sentence_representation = hidden_states[
            batch_indices,
            last_token_indices
        ]

        # Shape:
        # [batch_size, 768]


        # ---------------------------------------------
        # 4. Predict class
        # ---------------------------------------------

        logits = self.classifier(
            sentence_representation
        )

        # Shape:
        # [batch_size, num_classes]

        return logits
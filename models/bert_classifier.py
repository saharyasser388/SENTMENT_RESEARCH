"""BERT-base classifier used in Study 2."""
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, model_name="google-bert/bert-base-uncased", num_classes=7):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(output.last_hidden_state[:, 0])

"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: task model file.
"""


from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch


class BertForSequenceClassification(BertPreTrainedModel):
    """Sequence classification model,inherit from BertPreTrainedModel.
    
    @config: BertConfig class
    """
    def __init__(self, config: BertConfig):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids: list, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None) -> torch.Tensor:
        """Model forward propagation process, output model prediction results or loss.

        @input_ids: /
        """
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)      
        
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        else:
            outputs = nn.Softmax(dim=1)(logits)
        return outputs
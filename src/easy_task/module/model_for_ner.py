"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-09-23
@description: sequence tagging task model file.
"""


from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch


class BertForSequenceTagging(BertPreTrainedModel):
    """Sequence tagging model, inherit from BertPreTrainedModel.
    
    @config: BertConfig class
    """
    def __init__(self, config: BertConfig):
        super(BertForSequenceTagging, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.BIO_classifier = nn.Linear(config.hidden_size, config.num_labels)
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        BIO_logits = self.BIO_classifier(sequence_output)      
        
        if labels is not None:
            BIO_labels = labels
            BIO_loss = nn.CrossEntropyLoss(ignore_index=-1)(BIO_logits.view(-1, self.num_labels), BIO_labels.view(-1))
            outputs = BIO_loss
        else:
            outputs = nn.Softmax(dim=-1)(BIO_logits)
        return outputs


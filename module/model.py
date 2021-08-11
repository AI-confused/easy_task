# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F

class BertForSentenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSentenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.BIO_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[1]

        # predict BIO
        sequence_output = self.dropout(sequence_output)
        BIO_logits = self.BIO_classifier(sequence_output)      
        
        if labels is not None:
            BIO_labels = labels
            BIO_loss = nn.CrossEntropyLoss(ignore_index=-1)(BIO_logits.view(-1, self.num_labels), BIO_labels.view(-1))
            outputs = BIO_loss
        else:
            outputs = BIO_logits
        return outputs
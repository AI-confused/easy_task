"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-09-23
@description: task level function file.
"""

import torch.nn as nn
from collections import defaultdict
from base_module.base_utils import *
from base_module.base_result import *


class SequenceTaggingResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name.
    @id2label: id to label map dict.
    @max_seq_len: max sequence length of input feature.
    """
    def __init__(self, task_name: str, id2label: dict, max_seq_len: int):
        super(SequenceTaggingResult, self).__init__(task_name=task_name)
        self.bad_case = {'text': [], 'id': [], 'pred': [], 'label': []}
        self.all_result = {'text': [], 'id': [], 'pred': [], 'label': []}
        self.TP = 0
        self.Predict_num = 0
        self.Label_num = 0
        self.prediction = {}
        self.id2label = id2label
        self.max_seq_len = max_seq_len
        

    def get_entity_span(self, label: int, start: int, logit: int):
        """Get entity span end index.

        @label:
        @start:
        @logit:
        """
        index = start+1
        while index<len(logit):
            if logit[index] == label+len(self.id2label)-1:
                index += 1
            else:
                return index
        return index

        
    def decode_entity(self, logit: torch.Tensor, text: str):
        """Decode entitys from output logits and input texts.

        @logit: sequence tagging output of an input.
        @text: input sequence.
        """
        logit = logit.numpy().tolist()
        assert len(logit)==len(text)
        ans = []
        i = 0
        while i<len(logit):
            if logit[i] in range(1, len(self.id2label)):
                j = self.get_entity_span(logit[i], i, logit)
                key = self.id2label[logit[i]]
                value = text[i:j]
                i = j
                ans.append({key:value})
            else:
                i += 1
        return ans

    
    def reform_entity(self, entities: list):
        """Reform the format of pred entity.

        @entities: 
        """
        arguments_d = {}
        for d in entities:
            k = list(d.keys())[0]
            v = list(d.values())[0]
            if k not in arguments_d.keys():
                arguments_d[k] = defaultdict(int)
            arguments_d[k][v] += 1
        res = []
        for entity_type, entity_info in arguments_d.items():
            for entity, num in entity_info.items():
                for _ in range(num):
                    res.append({entity_type: entity})
        return res
        
        
    def update_batch(self, batch_results: list, **kwargs):
        """Update batch data in custom task.

        @batch_results: [batch_logits, batch_labels, batch_features]
        """
        batch_logits, _, batch_features = batch_results
        for i in range(len(batch_features)):
            example = batch_features[i]
            BIO_index = torch.max(batch_logits[i], dim=1)[1]
            assert len(BIO_index)==self.max_seq_len

            entities = self.decode_entity(BIO_index[1:example.sentence_len+1], example.sentence)

            if example.doc_id not in self.prediction.keys():
                self.prediction[example.doc_id] = {}
            if 'entity' not in self.prediction[example.doc_id].keys():
                self.prediction[example.doc_id]['entity'] = []
            self.prediction[example.doc_id]['entity'] += entities[:]

            try:
                if 'entity_label' not in self.prediction[example.doc_id].keys():
                    entity_labels = []
                    for entity_item in example.entity_label:
                        entity_labels.append({entity_item[0]: example.sentence[entity_item[1]:entity_item[2]]})
                    self.prediction[example.doc_id]['entity_label'] = entity_labels[:]
            except:
                pass

            if 'feature' not in self.prediction[example.doc_id].keys():
                self.prediction[example.doc_id]['feature'] = example
            
            
    def handle_pred_label(self, value: dict):
        """Handle prediction and label of one feature.

        @value: dict of one feature's pred and label.
        """
        # obtain label entity
        res_label = value['entity_label']

        # obtain prediction entity
        res_pred = []
        if value['entity']:
            res_pred = self.reform_entity(value['entity'])

        # return bad case
        if res_label != res_pred:
            self.bad_case['text'].append(value['feature'].sentence)
            self.bad_case['id'].append(value['feature'].doc_id)
            self.bad_case['pred'].append(res_pred[:])
            self.bad_case['label'].append(res_label[:])

        # return all prediction
        self.all_result['text'].append(value['feature'].sentence)
        self.all_result['id'].append(value['feature'].doc_id)
        self.all_result['pred'].append(res_pred[:])
        self.all_result['label'].append(res_label[:])

        return res_pred, res_label


    def get_score(self):
        """Calculate NER micro f1 score.
        """
        for _, value in self.prediction.items():
            pred, label = self.handle_pred_label(value)
            
            self.Label_num += len(label)
            self.Predict_num += len(pred)
            
            for item in label:
                if item in pred:
                    self.TP += 1
                    index = pred.index(item)
                    pred.pop(index)
            
        precision = self.TP/(self.Predict_num+1e-6)
        recall = self.TP/(self.Label_num+1e-6)
        micro_f1 = 2*precision*recall/(precision+recall+1e-6)
        
        return {'micro_f1': micro_f1, 'precision': precision, 'recall': recall}


    def get_prediction(self):
        """Obtain predictions in test-mode.
        """
        for _, value in self.prediction.items():
            res_pred = []
            if value['entity']:
                res_pred = self.reform_entity(value['entity'])

                # return prediction
                self.all_result['text'].append(value['feature'].sentence)
                self.all_result['id'].append(value['feature'].doc_id)
                self.all_result['pred'].append(res_pred[:])

        self.all_result.pop('label')
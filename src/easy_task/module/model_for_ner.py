"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-09-23
@description: sequence tagging task model file.
"""


from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch.autograd as autograd
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


# class BiLSTM_CRF(nn.Module):
#     """Sequence tagging model, BiLSTM_CRF layer.

#     @vocab_size: 
#     @tag_to_ix:
#     """
#     def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
#         super(BiLSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)

#         self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
#                             num_layers=1, bidirectional=True)

#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning *to* i *from* j.
#         self.transitions = nn.Parameter(
#             torch.randn(self.tagset_size, self.tagset_size))

#         # These two statements enforce the constraint that we never transfer
#         # to the start tag and we never transfer from the stop tag 
#         self.transitions.data[tag_to_ix[START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

#         self.hidden = self.init_hidden()

#     def init_hidden(self):
#         return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
#                 autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

#     def _forward_alg(self, feats):
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
#         # START_TAG has all of the score.
#         init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = autograd.Variable(init_alphas)
#         if CUDA_VALID:
#             forward_var = forward_var.cuda()
#         # Iterate through the sentence
#         for feat in feats:
#             alphas_t = []  # The forward variables at this timestep
#             for next_tag in range(self.tagset_size):
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(log_sum_exp(next_tag_var))
#             forward_var = torch.cat(alphas_t).view(1, -1)
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
#         alpha = log_sum_exp(terminal_var)
#         return alpha

#     def _get_lstm_features(self, sentence):
#         self.hidden = self.init_hidden()
#         embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
#         if CUDA_VALID:
#             self.hidden = tuple([elem.cuda() for elem in self.hidden])# Default is not cuda
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)
#         lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
#         lstm_feats = self.hidden2tag(lstm_out)
#         return lstm_feats

#     def _score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence
        
#         if CUDA_VALID:
#             tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).cuda(), tags])
#             score = autograd.Variable(torch.Tensor([0])).cuda()
#         else:
#             score = autograd.Variable(torch.Tensor([0]))
#             tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            
#         for i, feat in enumerate(feats):
#             score = score + \
#                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
#         score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
#         return score

#     def _viterbi_decode(self, feats):   
#         # Move feats from GPU to CPU
#         feats = feats.cpu()
#         backpointers = []

#         # Initialize the viterbi variables in log space
#         init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
#         init_vvars[0][self.tag_to_ix[START_TAG]] = 0

#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = autograd.Variable(init_vvars)
#         for feat in feats:
#             bptrs_t = []  # holds the backpointers for this step
#             viterbivars_t = []  # holds the viterbi variables for this step

#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
                
                
#                 # align data
#                 if CUDA_VALID:
#                     next_tag_var = forward_var + self.transitions[next_tag].cpu().view(1, -1)
#                 else:
#                     next_tag_var = forward_var + self.transitions[next_tag]

#                 best_tag_id = argmax(next_tag_var)
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id])
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
#             backpointers.append(bptrs_t)

#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].cpu().view(1,-1)
#         #terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
#         best_tag_id = argmax(terminal_var)
#         path_score = terminal_var[0][best_tag_id]

#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.tag_to_ix[START_TAG]  # Sanity check
#         best_path.reverse()
#         return path_score, best_path

#     def neg_log_likelihood(self, sentence, tags):
#         feats = self._get_lstm_features(sentence)
#         forward_score = self._forward_alg(feats)
#         gold_score = self._score_sentence(feats, tags)
#         return forward_score - gold_score

#     def forward(self, sentence):  # dont confuse this with _forward_alg above.
#         # Get the emission scores from the BiLSTM
#         lstm_feats = self._get_lstm_features(sentence)

#         # Find the best path, given the features.
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq
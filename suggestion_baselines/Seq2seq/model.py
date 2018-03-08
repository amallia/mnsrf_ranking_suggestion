###############################################################################
# Author: Wasi Ahmad
# Project: Sequence to sequence query generation
# Date Created: 06/04/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, Encoder, Decoder, AttentionDecoder


class Seq2Seq(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Seq2Seq, self).__init__()
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1

        self.embedding = EmbeddingLayer(len(dictionary), self.config)
        self.embedding.init_embedding_weights(dictionary, embedding_index, self.config.emsize)

        self.encoder = Encoder(self.config.emsize, self.config.nhid_enc, self.config.bidirection, self.config)
        if self.config.attn_type:
            self.decoder = AttentionDecoder(self.config.emsize, self.config.nhid_enc * self.num_directions,
                                            len(dictionary), self.config)
        else:
            self.decoder = Decoder(self.config.emsize, self.config.nhid_enc * self.num_directions, len(dictionary),
                                   self.config)

    @staticmethod
    def compute_decoding_loss(logits, target, seq_idx, length, regularize):
        losses = -torch.gather(logits, dim=1, index=target.unsqueeze(1)).squeeze()
        mask = helper.mask(length, seq_idx)  # mask: batch x 1
        losses = losses * mask.float()
        num_non_zero_elem = torch.nonzero(mask.data).size()
        if regularize:
            regularized_loss = logits.exp().mul(logits).sum(1).squeeze() * regularize
            loss = losses.sum() + regularized_loss.sum()
            if not num_non_zero_elem:
                return loss, 0
            else:
                return loss, num_non_zero_elem[0]
        else:
            if not num_non_zero_elem:
                return losses.sum(), 0
            else:
                return losses.sum(), num_non_zero_elem[0]

    def forward(self, q1_var, q1_len, q2_var, q2_len):
        # encode the query
        embedded_q1 = self.embedding(q1_var)
        encoded_q1, hidden = self.encoder(embedded_q1, q1_len)

        if self.config.bidirection:
            if self.config.model == 'LSTM':
                h_t, c_t = hidden[0][-2:], hidden[1][-2:]
                decoder_hidden = torch.cat((h_t[0].unsqueeze(0), h_t[1].unsqueeze(0)), 2), torch.cat(
                    (c_t[0].unsqueeze(0), c_t[1].unsqueeze(0)), 2)
            else:
                h_t = hidden[0][-2:]
                decoder_hidden = torch.cat((h_t[0].unsqueeze(0), h_t[1].unsqueeze(0)), 2)
        else:
            if self.config.model == 'LSTM':
                decoder_hidden = hidden[0][-1], hidden[1][-1]
            else:
                decoder_hidden = hidden[-1]

        if self.config.attn_type:
            decoder_context = Variable(torch.zeros(encoded_q1.size(0), encoded_q1.size(2))).unsqueeze(1)
            if self.config.cuda:
                decoder_context = decoder_context.cuda()

        decoding_loss, total_local_decoding_loss_element = 0, 0
        for idx in range(q2_var.size(1) - 1):
            input_variable = q2_var[:, idx]
            embedded_decoder_input = self.embedding(input_variable).unsqueeze(1)
            if self.config.attn_type:
                decoder_output, decoder_hidden, decoder_context, attn_weights = self.decoder(embedded_decoder_input,
                                                                                             decoder_hidden,
                                                                                             decoder_context,
                                                                                             encoded_q1)
            else:
                decoder_output, decoder_hidden = self.decoder(embedded_decoder_input, decoder_hidden)

            local_loss, num_local_loss = self.compute_decoding_loss(decoder_output, q2_var[:, idx + 1], idx, q2_len,
                                                                    self.config.regularize)
            decoding_loss += local_loss
            total_local_decoding_loss_element += num_local_loss

        if total_local_decoding_loss_element > 0:
            decoding_loss = decoding_loss / total_local_decoding_loss_element

        return decoding_loss

###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Framework
# Date Created: 7/15/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as f
from nn_layer import EmbeddingLayer, Encoder, EncoderCell, DecoderCell


class NSRF(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(NSRF, self).__init__()
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1

        self.embedding = EmbeddingLayer(len(dictionary), self.config)
        self.embedding.init_embedding_weights(dictionary, embedding_index, self.config.emsize)

        self.query_encoder = Encoder(self.config.emsize, self.config.nhid_query, self.config.bidirection, self.config)
        self.document_encoder = Encoder(self.config.emsize, self.config.nhid_doc, self.config.bidirection, self.config)
        self.session_query_encoder = EncoderCell(self.config.nhid_query * self.num_directions, self.config.nhid_session,
                                                 False, self.config)
        self.projection = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(
                self.config.nhid_query * self.num_directions + self.config.nhid_session,
                self.config.nhid_doc * self.num_directions)),
            ('tanh', nn.Tanh())
        ]))
        self.decoder = DecoderCell(self.config.emsize, self.config.nhid_session, len(dictionary), self.config)

    def forward(self, session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels):
        """
        Forward function of the neural click model. Return average loss for a batch of sessions.
        :param session_queries: 3d tensor [batch_size x session_length x max_query_length]
        :param session_query_length: 2d tensor [batch_size x session_length]
        :param rel_docs: 4d tensor [batch_size x session_length x num_rel_docs_per_query x max_doc_length]
        :param rel_docs_length: 3d tensor [batch_size x session_length x num_rel_docs_per_query]
        :param doc_labels: 3d tensor [batch_size x session_length x num_rel_docs_per_query]
        :return: average loss over batch [autograd Variable]
        """
        # query encoding
        embedded_queries = self.embedding(session_queries.view(-1, session_queries.size(-1)))
        encoded_queries = self.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
        encoded_queries = self.apply_pooling(encoded_queries, self.config.pool_type)
        # encoded_queries: batch_size x session_length x (nhid_query * self.num_directions)
        encoded_queries = encoded_queries.view(*session_queries.size()[:-1], -1)

        # document encoding
        embedded_docs = self.embedding(rel_docs.view(-1, rel_docs.size(-1)))
        encoded_docs = self.document_encoder(embedded_docs, rel_docs_length.view(-1).data.cpu().numpy())
        encoded_docs = self.apply_pooling(encoded_docs, self.config.pool_type)
        # encoded_docs: batch_size x session_length x num_rel_docs_per_query x (nhid_doc * self.num_directions)
        encoded_docs = encoded_docs.view(*rel_docs.size()[:-1], -1)

        # session level encoding
        sess_q_hidden = self.session_query_encoder.init_weights(encoded_queries.size(0))
        sess_q_out = Variable(torch.zeros(session_queries.size(0), 1, self.config.nhid_session))
        if self.config.cuda:
            sess_q_out = sess_q_out.cuda()

        hidden_states, cell_states = [], []
        click_loss = 0
        # loop over all the queries in a session
        for idx in range(encoded_queries.size(1)):
            combined_rep = torch.cat((encoded_queries[:, idx, :], sess_q_out.squeeze(1)), 1)
            combined_rep = self.projection(combined_rep)
            combined_rep = combined_rep.unsqueeze(1).expand(*encoded_docs[:, idx, :, :].size())
            click_score = torch.sum(torch.mul(combined_rep, encoded_docs[:, idx, :, :]), 2)
            click_loss += f.binary_cross_entropy_with_logits(click_score, doc_labels[:, idx, :])
            # update session-level query encoder state using query representations
            sess_q_out, sess_q_hidden = self.session_query_encoder(encoded_queries[:, idx, :].unsqueeze(1),
                                                                   sess_q_hidden)
            # -1: only consider hidden states of the last layer
            if self.config.model == 'LSTM':
                hidden_states.append(sess_q_hidden[0][-1])
                cell_states.append(sess_q_hidden[1][-1])
            else:
                hidden_states.append(sess_q_hidden[-1])

        click_loss = click_loss / encoded_queries.size(1)
        hidden_states = torch.stack(hidden_states, 1)
        # remove the last hidden states which stand for the last queries in sessions
        hidden_states = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1)).unsqueeze(0)
        if self.config.model == 'LSTM':
            cell_states = torch.stack(cell_states, 1)
            cell_states = cell_states[:, :-1, :].contiguous().view(-1, cell_states.size(-1)).unsqueeze(0)
            # Initialize hidden states of decoder with the last hidden states of the session encoder
            decoder_hidden = (hidden_states, cell_states)
        else:
            # Initialize hidden states of decoder with the last hidden states of the session encoder
            decoder_hidden = hidden_states

        # train the decoder for all the queries in a session except the last
        embedded_queries = embedded_queries.view(*session_queries.size(), -1)
        decoder_input = embedded_queries[:, 1:, :, :].contiguous().view(-1, *embedded_queries.size()[2:])
        decoder_target = session_queries[:, 1:, :].contiguous().view(-1, session_queries.size(-1))
        target_length = session_query_length[:, 1:].contiguous().view(-1)

        decoding_loss, total_local_decoding_loss_element = 0, 0
        for idx in range(decoder_input.size(1) - 1):
            input_variable = decoder_input[:, idx, :].unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(input_variable, decoder_hidden)
            local_loss, num_local_loss = self.compute_decoding_loss(decoder_output, decoder_target[:, idx + 1], idx,
                                                                    target_length, self.config.regularize)
            decoding_loss += local_loss
            total_local_decoding_loss_element += num_local_loss

        if total_local_decoding_loss_element > 0:
            decoding_loss = decoding_loss / total_local_decoding_loss_element

        return click_loss + decoding_loss

    @staticmethod
    def apply_pooling(encodings, pool_type):
        if pool_type == 'max':
            pooled_encodings = torch.max(encodings, 1)[0].squeeze()
        elif pool_type == 'mean':
            pooled_encodings = torch.sum(encodings, 1).squeeze() / encodings.size(1)
        elif pool_type == 'last':
            pooled_encodings = encodings[:, -1, :]

        return pooled_encodings

    @staticmethod
    def compute_decoding_loss(logits, target, seq_idx, length, regularize):
        """
        Compute negative log-likelihood loss for a batch of predictions.
        :param logits: 2d tensor [batch_size x vocab_size]
        :param target: 1d tensor [batch_size]
        :param seq_idx: an integer represents the current index of the sequences
        :param length: 1d tensor [batch_size], represents each sequences' true length
        :return: total loss over the input mini-batch [autograd Variable] and number of loss elements
        """
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

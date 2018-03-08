###############################################################################
# Author: Wasi Ahmad
# Project: Multi-task Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/29/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from nn_layer import EmbeddingLayer, Encoder, ExactMatchChannel, EncoderCell, DecoderCell


class MatchTensor(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(MatchTensor, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)
        self.linear_projection = nn.Linear(self.config.emsize, self.config.featsize)
        self.query_encoder = Encoder(self.config.featsize, self.config.nhid_query, True, self.config)
        self.document_encoder = Encoder(self.config.featsize, self.config.nhid_doc, True, self.config)
        self.query_projection = nn.Linear(self.config.nhid_query * self.num_directions, self.config.nchannels)
        self.document_projection = nn.Linear(self.config.nhid_doc * self.num_directions, self.config.nchannels)

        self.exact_match_channel = ExactMatchChannel()
        self.conv1 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 7), padding=(1, 3))
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(self.config.nfilters * 3, self.config.match_filter_size, (1, 1))
        self.output = nn.Linear(self.config.match_filter_size, 1)

        self.session_encoder = EncoderCell(self.config.nchannels, self.config.nhid_session, False, self.config)
        self.decoder = DecoderCell(self.config.emsize, self.config.nhid_session, len(dictionary), self.config)

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
        batch_queries = session_queries.view(-1, session_queries.size(-1))
        batch_docs = rel_docs.view(-1, *rel_docs.size()[2:])

        projected_queries = self.encode_query(batch_queries, session_query_length)  # (B*S) x L x H
        projected_docs = self.encode_document(batch_docs, rel_docs_length)
        score = self.document_ranker(projected_queries, projected_docs, batch_queries, batch_docs)
        click_loss = f.binary_cross_entropy_with_logits(score, doc_labels.view(-1, doc_labels.size(2)))

        # encoded_queries: batch_size x session_length x nhid_query
        encoded_queries = projected_queries.max(1)[0].view(*session_queries.size()[:2], -1)
        decoding_loss = self.query_recommender(session_queries, session_query_length, encoded_queries)

        return click_loss, decoding_loss

    def query_recommender(self, session_queries, session_query_length, encoded_queries):
        # session level encoding
        sess_q_hidden = self.session_encoder.init_weights(encoded_queries.size(0))
        hidden_states, cell_states = [], []
        # loop over all the queries in a session
        for idx in range(encoded_queries.size(1)):
            # update session-level query encoder state using query representations
            sess_q_out, sess_q_hidden = self.session_encoder(encoded_queries[:, idx, :].unsqueeze(1), sess_q_hidden)
            # -1 stands for: only consider hidden states from the last layer
            if self.config.model == 'LSTM':
                hidden_states.append(sess_q_hidden[0][-1])
                cell_states.append(sess_q_hidden[1][-1])
            else:
                hidden_states.append(sess_q_hidden[-1])

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

        embedded_queries = self.embedding(session_queries.view(-1, session_queries.size(-1)))
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

        return decoding_loss

    def document_ranker(self, projected_queries, projected_docs, batch_queries, batch_docs):
        # step6: 2d product between projected query and doc vectors
        projected_queries = projected_queries.unsqueeze(1).expand(projected_queries.size(0), batch_docs.size(1),
                                                                  *projected_queries.size()[1:])
        projected_queries = projected_queries.contiguous().view(-1, *projected_queries.size()[2:])

        projected_docs = projected_docs.view(-1, batch_docs.size(2), projected_docs.size()[-1])

        projected_queries = projected_queries.unsqueeze(2).expand(*projected_queries.size()[:2], batch_docs.size()[-1],
                                                                  projected_queries.size(2))
        projected_docs = projected_docs.unsqueeze(1).expand(projected_docs.size(0), batch_queries.size()[-1],
                                                            *projected_docs.size()[1:])
        query_document_product = projected_queries * projected_docs

        # step7: append exact match channel
        exact_match = self.exact_match_channel(batch_queries, batch_docs).unsqueeze(3)
        query_document_product = torch.cat((query_document_product, exact_match), 3)
        query_document_product = query_document_product.transpose(2, 3).transpose(1, 2)

        # step8: run the convolutional operation, max-pooling and linear projection
        convoluted_feat1 = self.conv1(query_document_product)
        convoluted_feat2 = self.conv2(query_document_product)
        convoluted_feat3 = self.conv3(query_document_product)
        convoluted_feat = self.relu(torch.cat((convoluted_feat1, convoluted_feat2, convoluted_feat3), 1))
        convoluted_feat = self.conv(convoluted_feat).transpose(1, 2).transpose(2, 3)

        max_pooled_feat = torch.max(convoluted_feat, 2)[0].squeeze()
        max_pooled_feat = torch.max(max_pooled_feat, 1)[0].squeeze()
        return self.output(max_pooled_feat).squeeze().view(*batch_docs.size()[:2])

    def encode_query(self, batch_queries, session_query_length):
        # step1: apply embedding lookup
        embedded_queries = self.embedding(batch_queries)
        # step2: apply linear projection on embedded queries and documents
        embedded_queries = self.linear_projection(embedded_queries.view(-1, embedded_queries.size(-1)))
        # step3: transform the tensors so that they can be given as input to RNN
        embedded_queries = embedded_queries.view(*batch_queries.size(), self.config.featsize)
        # step4: pass the encoded query and doc through a bi-LSTM
        encoded_queries = self.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
        # step5: apply linear projection on query hidden states
        projected_queries = self.query_projection(encoded_queries.view(-1, encoded_queries.size()[-1])).view(
            *batch_queries.size(), -1)
        return projected_queries

    def encode_document(self, batch_docs, rel_docs_length):
        # step1: apply embedding lookup
        embedded_docs = self.embedding(batch_docs.view(-1, batch_docs.size(-1)))
        # step2: apply linear projection on embedded queries and documents
        embedded_docs = self.linear_projection(embedded_docs.view(-1, embedded_docs.size(-1)))
        # step3: transform the tensors so that they can be given as input to RNN
        embedded_docs = embedded_docs.view(-1, batch_docs.size()[-1], self.config.featsize)
        # step4: pass the encoded query and doc through a bi-LSTM
        encoded_docs = self.document_encoder(embedded_docs, rel_docs_length.view(-1).data.cpu().numpy())
        # step5: apply linear projection on query hidden states
        projected_docs = self.document_projection(encoded_docs.view(-1, encoded_docs.size()[-1]))
        return projected_docs

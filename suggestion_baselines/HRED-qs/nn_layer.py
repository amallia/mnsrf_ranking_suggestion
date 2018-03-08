###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script contains code related to the neural network layers.
###############################################################################

import torch, helper
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as f


class EmbeddingLayer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(EmbeddingLayer, self).__init__()

        if config.emtraining:
            self.embedding = nn.Sequential(OrderedDict([
                ('embedding', nn.Embedding(input_size, config.emsize)),
                ('dropout', nn.Dropout(config.dropout))
            ]))
        else:
            self.embedding = nn.Embedding(input_size, config.emsize)
            self.embedding.weight.requires_grad = False

    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        return self.embedding(input_variable)

    def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embeddings_index:
                pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i] = helper.initialize_out_of_vocab_words(embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        if isinstance(self.embedding, nn.Sequential):
            self.embedding[0].weight.data.copy_(torch.from_numpy(pretrained_weight))
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirection = bidirection

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.bidirection)

    def forward(self, sent_variable, sent_len):
        """"Defines the forward computation of the encoder"""
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.config.cuda else torch.from_numpy(idx_sort)
        sent_variable = sent_variable.index_select(0, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len, batch_first=True)
        sent_output = self.rnn(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.config.cuda else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(0, Variable(idx_unsort))

        return sent_output


class EncoderCell(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, config):
        """"Constructor of the class"""
        super(EncoderCell, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirection = bidirection

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.bidirection)

    def forward(self, input, hidden):
        """"Defines the forward computation of the encoder"""
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirection else 1
        if self.config.model == 'LSTM':
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_()), Variable(
                weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_())
        else:
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_())


class DecoderCell(nn.Module):
    """Decoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, output_size, config):
        """"Constructor of the class"""
        super(DecoderCell, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, output_size)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, 1, batch_first=True,
                                                      dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, 1, nonlinearity=nonlinearity, batch_first=True,
                              dropout=self.config.dropout)

    def forward(self, input, hidden):
        """"Defines the forward computation of the decoder"""
        output, hidden = self.rnn(input, hidden)
        output = f.log_softmax(self.out(output.squeeze(1)), 1)
        return output, hidden

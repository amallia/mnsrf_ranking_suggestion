###############################################################################
# Author: Wasi Ahmad
# Project: ARC-I: Convolutional Matching Model
# Date Created: 7/28/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_layer import EmbeddingLayer


class CNN_ARC_I(nn.Module):
    """Implementation of the convolutional matching model (ARC-II)."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(CNN_ARC_I, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.convolution1 = nn.Conv1d(self.config.emsize, self.config.nfilters, 1)
        self.convolution2 = nn.Conv1d(self.config.emsize, self.config.nfilters, 2)
        self.convolution3 = nn.Conv1d(self.config.emsize, self.config.nfilters * 2, 3)
        self.ffnn = nn.Sequential(nn.Linear(self.config.nfilters * 8, self.config.nfilters * 4),
                                  nn.Linear(self.config.nfilters * 4, self.config.nfilters * 2),
                                  nn.Linear(self.config.nfilters * 2, 1))

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    def forward(self, batch_queries, batch_docs):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :return: click probabilities [batch_size x num_rel_docs_per_query]
        """
        embedded_queries = self.embedding(batch_queries)
        embedded_docs = self.embedding(batch_docs.view(-1, batch_docs.size(-1)))

        convolved_query_1 = self.convolution1(embedded_queries.transpose(1, 2)).transpose(1, 2)
        max_pooled_query_1 = torch.max(convolved_query_1, 1)[0].squeeze()
        convolved_doc_1 = self.convolution1(embedded_docs.transpose(1, 2)).transpose(1, 2)
        max_pooled_doc_1 = torch.max(convolved_doc_1, 1)[0].squeeze()

        convolved_query_2 = self.convolution2(embedded_queries.transpose(1, 2)).transpose(1, 2)
        max_pooled_query_2 = torch.max(convolved_query_2, 1)[0].squeeze()
        convolved_doc_2 = self.convolution2(embedded_docs.transpose(1, 2)).transpose(1, 2)
        max_pooled_doc_2 = torch.max(convolved_doc_2, 1)[0].squeeze()

        convolved_query_3 = self.convolution3(embedded_queries.transpose(1, 2)).transpose(1, 2)
        max_pooled_query_3 = torch.max(convolved_query_3, 1)[0].squeeze()
        convolved_doc_3 = self.convolution3(embedded_docs.transpose(1, 2)).transpose(1, 2)
        max_pooled_doc_3 = torch.max(convolved_doc_3, 1)[0].squeeze()

        query_rep = torch.cat((max_pooled_query_1, max_pooled_query_2, max_pooled_query_3), 1).unsqueeze(1)
        query_rep = query_rep.expand(*batch_docs.size()[0:2], query_rep.size(2))
        query_rep = query_rep.contiguous().view(-1, query_rep.size(2))
        doc_rep = torch.cat((max_pooled_doc_1, max_pooled_doc_2, max_pooled_doc_3), 1)

        combined_representation = torch.cat((query_rep, doc_rep), 1)
        return F.log_softmax(self.ffnn(combined_representation).squeeze().view(*batch_docs.size()[0:2]), 1)

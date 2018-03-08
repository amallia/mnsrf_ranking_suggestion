###############################################################################
# Author: Wasi Ahmad
# Project: ARC-II: Convolutional Matching Model
# Date Created: 7/28/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_layer import EmbeddingLayer


class CNN_ARC_II(nn.Module):
    """Implementation of the convolutional matching model (ARC-II)."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(CNN_ARC_II, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.conv1 = nn.Conv2d(self.config.emsize * 2, self.config.nfilters, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(self.config.nfilters, self.config.nfilters, (2, 2))
        self.ffnn = nn.Sequential(nn.Linear(self.config.nfilters * 4, self.config.nfilters * 2),
                                  nn.Linear(self.config.nfilters * 2, self.config.nfilters),
                                  nn.Linear(self.config.nfilters, 1))

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    def forward(self, batch_queries, batch_docs):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :return: score representing click probability [batch_size x num_clicks_per_query]
        """
        embedded_queries = self.embedding(batch_queries)
        embedded_queries = embedded_queries.unsqueeze(1).expand(*batch_docs.size()[:2], *embedded_queries.size()[1:])
        embedded_queries = embedded_queries.contiguous().view(-1, *embedded_queries.size()[2:])
        embedded_docs = self.embedding(batch_docs.view(-1, batch_docs.size(-1)))

        embedded_queries = embedded_queries.unsqueeze(1).expand(embedded_queries.size(0), batch_docs.size(2),
                                                                *embedded_queries.size()[1:])
        embedded_docs = embedded_docs.unsqueeze(2).expand(*embedded_docs.size()[:2], batch_queries.size(1),
                                                          embedded_docs.size(2))

        combined_rep = torch.cat((embedded_queries, embedded_docs), 3)
        combined_rep = combined_rep.transpose(2, 3).transpose(1, 2)
        conv1_out = self.pool1(F.relu(self.conv1(combined_rep)))
        conv2_out = self.pool1(F.relu(self.conv2(conv1_out))).squeeze().view(-1, self.config.nfilters * 4)

        return F.log_softmax(self.ffnn(conv2_out).squeeze().view(*batch_docs.size()[0:2]), 1)

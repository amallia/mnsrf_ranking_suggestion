###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, numpy
import torch.nn as nn
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, GatingNetwork


class DRMM(nn.Module):
    """Implementation of the deep relevance matching model."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(DRMM, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.bins = [-1.0, -0.5, 0, 0.5, 1.0, 1.0]

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.gating_network = GatingNetwork(self.config.emsize)
        self.ffnn = nn.Sequential(nn.Linear(self.config.nbins, 1), nn.Linear(1, 1))
        self.output = nn.Linear(1, 1)

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """
        Returns cosine similarity between x1 and x2, computed along dim.
        # taken from http://pytorch.org/docs/master/_modules/torch/nn/functional.html#cosine_similarity
        :param x1: (Variable): First input.
        :param x2: (Variable): Second input (of size matching x1).
        :param dim: (int, optional): Dimension of vectors. Default: 1
        :param eps: Small value to avoid division by zero. Default: 1e-8
        :return: 
        """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    def forward(self, batch_queries, batch_docs):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :return: score representing click probability [batch_size x num_clicks_per_query]
        """
        embedded_queries = self.embedding(batch_queries)
        term_weights = self.gating_network(embedded_queries).unsqueeze(1).expand(*batch_docs.size()[:2],
                                                                                 batch_queries.size(1))
        embedded_docs = self.embedding(batch_docs.view(-1, batch_docs.size(-1)))
        embedded_queries = embedded_queries.unsqueeze(1).expand(*batch_docs.size()[:2], *embedded_queries.size()[1:])
        embedded_queries = embedded_queries.contiguous().view(-1, *embedded_queries.size()[2:])

        embedded_queries = embedded_queries.unsqueeze(2).expand(*embedded_queries.size()[:2], batch_docs.size(2),
                                                                embedded_queries.size(2))
        embedded_docs = embedded_docs.unsqueeze(1).expand(embedded_docs.size(0), batch_queries.size(1),
                                                          *embedded_docs.size()[1:])
        cos_sim = self.cosine_similarity(embedded_queries, embedded_docs, 3)

        hist = numpy.apply_along_axis(lambda x: numpy.histogram(x, bins=self.bins), 2, cos_sim.data.cpu().numpy())
        histogram_feats = torch.from_numpy(numpy.array([[axis2 for axis2 in axis1] for axis1 in hist[:, :, 0]])).float()
        if self.config.cuda:
            histogram_feats = Variable(histogram_feats).cuda()
        else:
            histogram_feats = Variable(histogram_feats)

        ffnn_out = self.ffnn(histogram_feats.view(-1, self.config.nbins)).squeeze().view(-1, batch_queries.size(1))
        weighted_ffnn_out = ffnn_out * term_weights.contiguous().view(-1, term_weights.size(2))
        score = self.output(torch.sum(weighted_ffnn_out, 1, keepdim=True)).squeeze()
        return score.view(*batch_docs.size()[:2])

###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
# Date Created: 7/18/2017
#
# File Description: This script implements the deep semantic similarity model.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f


class DSSM(nn.Module):
    """Implementation of the deep semantic similarity model."""

    def __init__(self, dictionary, args):
        """"Constructor of the class."""
        super(DSSM, self).__init__()
        self.dictionary = dictionary
        self.config = args
        self.generate_semantic_feature = nn.Sequential(nn.Linear(len(dictionary), self.config.nhid), nn.Tanh(),
                                                       nn.Linear(self.config.nhid, self.config.nhid), nn.Tanh(),
                                                       nn.Linear(self.config.nhid, self.config.nhid_output), nn.Tanh())

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
        Forward function of the dssm model. Return average loss for a batch of queries.
        :param batch_queries: 2d tensor [batch_size x vocab_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x vocab_size]
        :return: softmax score representing click probability [batch_size x num_rel_docs_per_query]
        """
        # query encoding
        query_rep = self.generate_semantic_feature(batch_queries)
        # document encoding
        doc_rep = self.generate_semantic_feature(batch_docs.view(-1, batch_docs.size(2)))
        doc_rep = doc_rep.view(*batch_docs.size()[:-1], -1)
        # compute loss
        query_rep = query_rep.unsqueeze(1).expand(*doc_rep.size())
        return f.log_softmax(self.cosine_similarity(query_rep, doc_rep, 2), 1)

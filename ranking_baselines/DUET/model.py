###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script implements the deep semantic similarity model.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f


# verified from https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb
class DUET(nn.Module):
    """Learning to Match using Local and Distributed Representations of Text for Web Search."""

    def __init__(self, dictionary, args):
        """"Constructor of the class."""
        super(DUET, self).__init__()
        self.dictionary = dictionary
        self.config = args
        self.local_model = LocalModel(self.config)
        self.distributed_model = DistributedModel(self.config, len(self.dictionary))

    def forward(self, batch_queries, batch_docs):
        """
        Forward function of the dssm model. Return average loss for a batch of queries.
        :param batch_queries: 2d tensor [batch_size x vocab_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x vocab_size]
        :return: softmax score representing click probability [batch_size x num_rel_docs_per_query]
        """
        local_score = self.local_model(batch_queries, batch_docs)
        distributed_score = self.distributed_model(batch_queries, batch_docs)
        total_score = local_score + distributed_score
        return f.log_softmax(total_score, 1)


class LocalModel(nn.Module):
    """Implementation of the local model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(LocalModel, self).__init__()
        self.config = args
        self.conv1d = nn.Conv1d(self.config.max_doc_length, self.config.nfilters, self.config.local_filter_size)
        self.drop = nn.Dropout(self.config.dropout)
        self.fc1 = nn.Linear(self.config.max_query_length, 1)
        self.fc2 = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.fc3 = nn.Linear(self.config.nfilters, 1)

    def forward(self, batch_queries, batch_clicks):
        output_size = batch_clicks.size()[:2]
        batch_queries = batch_queries.unsqueeze(1).expand(batch_queries.size(0), batch_clicks.size(1),
                                                          *batch_queries.size()[1:])
        batch_queries = batch_queries.contiguous().view(-1, *batch_queries.size()[2:]).float()
        batch_clicks = batch_clicks.view(-1, *batch_clicks.size()[2:]).transpose(1, 2).float()
        interaction_feature = torch.bmm(batch_queries, batch_clicks).transpose(1, 2)
        convolved_feature = self.conv1d(interaction_feature)
        mapped_feature1 = f.tanh(self.fc1(convolved_feature.view(-1, convolved_feature.size(2)))).squeeze(1)
        mapped_feature1 = mapped_feature1.view(*convolved_feature.size()[:-1])
        mapped_feature2 = self.drop(f.tanh(self.fc2(mapped_feature1)))
        score = f.tanh(self.fc3(mapped_feature2)).view(*output_size)
        return score


class DistributedModel(nn.Module):
    """Implementation of the distributed model."""

    def __init__(self, args, vocab_size):
        """"Constructor of the class."""
        super(DistributedModel, self).__init__()
        self.config = args
        self.conv1d = nn.Conv1d(vocab_size, self.config.nfilters, self.config.dist_filter_size)
        self.drop = nn.Dropout(self.config.dropout)
        self.fc1_query = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.conv1d_doc = nn.Conv1d(self.config.nfilters, self.config.nfilters, 1)
        self.fc2 = nn.Linear(self.config.max_doc_length - self.config.pool_size - 1, 1)
        self.fc3 = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.fc4 = nn.Linear(self.config.nfilters, 1)

    def forward(self, batch_queries, batch_clicks):
        output_size = batch_clicks.size()[:2]
        batch_queries = batch_queries.transpose(1, 2).float()
        batch_clicks = batch_clicks.view(-1, *batch_clicks.size()[2:]).transpose(1, 2).float()
        # apply convolution 1d
        convolved_query_features = self.conv1d(batch_queries)
        convolved_doc_features = self.conv1d(batch_clicks)
        # apply max-pooling 1d
        maxpooled_query_features = f.max_pool1d(convolved_query_features, convolved_query_features.size(2)).squeeze(2)
        maxpooled_doc_features = f.max_pool1d(convolved_doc_features, self.config.pool_size, 1)
        # apply fc to query and convolution 1d to document representation
        query_rep = f.tanh(self.fc1_query(maxpooled_query_features))
        doc_rep = self.conv1d_doc(maxpooled_doc_features)
        # do hadamard (element-wise) product
        query_rep = query_rep.unsqueeze(2).expand(*query_rep.size(), doc_rep.size(2)).unsqueeze(1)
        query_rep = query_rep.expand(query_rep.size(0), output_size[1], *query_rep.size()[2:])
        query_rep = query_rep.contiguous().view(-1, *query_rep.size()[2:])
        query_doc_sim = query_rep * doc_rep
        # apply fc2
        mapped_features = f.tanh(self.fc2(query_doc_sim.view(-1, query_doc_sim.size(2)))).squeeze(1)
        mapped_features = mapped_features.view(*query_doc_sim.size()[:-1])
        # apply fc3 and dropout
        mapped_features_2 = self.drop(f.tanh(self.fc3(mapped_features)))
        # apply fc4
        score = f.tanh(self.fc4(mapped_features_2)).view(*output_size)
        return score

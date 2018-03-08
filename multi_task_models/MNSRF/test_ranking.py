###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Framework
# Date Created: 7/15/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
from torch.autograd import Variable
from model import NSRF
from rank_metrics import mean_average_precision, NDCG, MRR

args = util.get_args()


def compute_ranking_performance(model, session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels):
    # query encoding
    embedded_queries = model.embedding(session_queries.view(-1, session_queries.size(-1)))
    encoded_queries = model.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
    encoded_queries = model.apply_pooling(encoded_queries, model.config.pool_type)
    # encoded_queries: batch_size x session_length x (nhid_query * self.num_directions)
    encoded_queries = encoded_queries.view(*session_queries.size()[:-1], -1)

    # document encoding
    embedded_docs = model.embedding(rel_docs.view(-1, rel_docs.size(-1)))
    encoded_docs = model.document_encoder(embedded_docs, rel_docs_length.view(-1).data.cpu().numpy())
    encoded_docs = model.apply_pooling(encoded_docs, model.config.pool_type)
    # encoded_docs: batch_size x session_length x num_rel_docs_per_query x (nhid_doc * self.num_directions)
    encoded_docs = encoded_docs.view(*rel_docs.size()[:-1], -1)

    # session level encoding
    sess_q_hidden = model.session_query_encoder.init_weights(encoded_queries.size(0))
    sess_q_out = Variable(torch.zeros(session_queries.size(0), 1, model.config.nhid_session))
    if model.config.cuda:
        sess_q_out = sess_q_out.cuda()

    map, mrr, NDCG_at_1, NDCG_at_3, NDCG_at_5, NDCG_at_10 = 0, 0, 0, 0, 0, 0
    # loop over all the queries in a session
    for idx in range(encoded_queries.size(1)):
        combined_rep = torch.cat((encoded_queries[:, idx, :], sess_q_out.squeeze(1)), 1)
        combined_rep = model.projection(combined_rep)
        combined_rep = combined_rep.unsqueeze(1).expand(*encoded_docs[:, idx, :, :].size())
        click_score = torch.sum(torch.mul(combined_rep, encoded_docs[:, idx, :, :]), 2)
        numpy_score, numpy_labels = click_score.data.cpu().numpy(), doc_labels[:, idx, :].data.cpu().numpy()

        map += mean_average_precision(numpy_score, numpy_labels)
        mrr += MRR(numpy_score, numpy_labels)
        NDCG_at_1 += NDCG(numpy_score, numpy_labels, 1)
        NDCG_at_3 += NDCG(numpy_score, numpy_labels, 3)
        NDCG_at_5 += NDCG(numpy_score, numpy_labels, 5)
        NDCG_at_10 += NDCG(numpy_score, numpy_labels, 10)
        # update session-level query encoder state using query representations
        sess_q_out, sess_q_hidden = model.session_query_encoder(encoded_queries[:, idx, :].unsqueeze(1),
                                                                sess_q_hidden)

    map = map / encoded_queries.size(1)
    mrr = mrr / encoded_queries.size(1)
    NDCG_at_1 = NDCG_at_1 / encoded_queries.size(1)
    NDCG_at_3 = NDCG_at_3 / encoded_queries.size(1)
    NDCG_at_5 = NDCG_at_5 / encoded_queries.size(1)
    NDCG_at_10 = NDCG_at_10 / encoded_queries.size(1)

    return map, mrr, NDCG_at_1, NDCG_at_3, NDCG_at_5, NDCG_at_10


def test_ranking(model, test_batches, dictionary):
    num_batches = len(test_batches)
    map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels = helper.session_to_tensor(
            test_batches[batch_no - 1], dictionary, iseval=True)
        if model.config.cuda:
            session_queries = session_queries.cuda()
            session_query_length = session_query_length.cuda()
            rel_docs = rel_docs.cuda()
            rel_docs_length = rel_docs_length.cuda()
            doc_labels = doc_labels.cuda()

        ret_val = compute_ranking_performance(model, session_queries, session_query_length, rel_docs, rel_docs_length,
                                              doc_labels)
        map += ret_val[0]
        mrr += ret_val[1]
        ndcg_1 += ret_val[2]
        ndcg_3 += ret_val[3]
        ndcg_5 += ret_val[4]
        ndcg_10 += ret_val[5]

    _map = map / num_batches
    mrr = mrr / num_batches
    ndcg_1 = ndcg_1 / num_batches
    ndcg_3 = ndcg_3 / num_batches
    ndcg_5 = ndcg_5 / num_batches
    ndcg_10 = ndcg_10 / num_batches

    print('MAP - ', _map)
    print('MRR - ', mrr)
    print('NDCG@1 - ', ndcg_1)
    print('NDCG@3 - ', ndcg_3)
    print('NDCG@5 - ', ndcg_5)
    print('NDCG@10 - ', ndcg_10)


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = NSRF(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()

    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('model, embedding index and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example)
    print('test set size = ', len(test_corpus))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('number of test batches = ', len(test_batches))
    test_ranking(model, test_batches, dictionary)

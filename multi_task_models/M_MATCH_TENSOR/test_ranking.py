###############################################################################
# Author: Wasi Ahmad
# Project: Multi-task Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/29/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
import numpy as np
from model import MatchTensor
from rank_metrics import mean_average_precision, NDCG, MRR

args = util.get_args()
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def compute_ranking_performance(model, session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels):
    batch_queries = session_queries.view(-1, session_queries.size(-1))
    batch_docs = rel_docs.view(-1, *rel_docs.size()[2:])

    projected_queries = model.encode_query(batch_queries, session_query_length)
    projected_docs = model.encode_document(batch_docs, rel_docs_length)
    score = model.document_ranker(projected_queries, projected_docs, batch_queries, batch_docs)

    numpy_score, numpy_labels = score.data.cpu().numpy(), doc_labels.view(-1, doc_labels.size(-1)).data.cpu().numpy()
    map = mean_average_precision(numpy_score, numpy_labels)
    mrr = MRR(numpy_score, numpy_labels)
    NDCG_at_1 = NDCG(numpy_score, numpy_labels, 1)
    NDCG_at_3 = NDCG(numpy_score, numpy_labels, 3)
    NDCG_at_5 = NDCG(numpy_score, numpy_labels, 5)
    NDCG_at_10 = NDCG(numpy_score, numpy_labels, 10)

    return map, mrr, NDCG_at_1, NDCG_at_3, NDCG_at_5, NDCG_at_10


def test_ranking(model, test_batches, dictionary):
    num_batches = len(test_batches)
    map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels = helper.session_to_tensor(
            test_batches[batch_no - 1], dictionary, True)
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
    model = MatchTensor(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('Model, embedding index and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example)
    print('test set size = ', len(test_corpus))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('number of test batches = ', len(test_batches))
    test_ranking(model, test_batches, dictionary)

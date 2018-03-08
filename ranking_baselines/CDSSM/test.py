###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os, numpy
from model import CDSSM
from rank_metrics import mean_average_precision, NDCG, MRR

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    _map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        test_queries, test_docs, test_labels = helper.batch_to_tensor(test_batches[batch_no - 1], model.dictionary)
        if model.config.cuda:
            test_queries = test_queries.cuda()
            test_docs = test_docs.cuda()
            test_labels = test_labels.cuda()

        softmax_prob = model(test_queries, test_docs)
        _map += mean_average_precision(softmax_prob, test_labels)
        mrr += MRR(softmax_prob, test_labels)
        ndcg_1 += NDCG(softmax_prob, test_labels, 1)
        ndcg_3 += NDCG(softmax_prob, test_labels, 3)
        ndcg_5 += NDCG(softmax_prob, test_labels, 5)
        ndcg_10 += NDCG(softmax_prob, test_labels, 10)

    _map = _map / num_batches
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
    model = CDSSM(dictionary, args)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(cuda_visible_devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
    if args.cuda:
        model = model.cuda()

    checkpoint = helper.load_from_checkpoint(os.path.join(args.save_path, 'model_best.pth.tar'), args.cuda)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example)
    print('test set size = ', len(test_corpus.data))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('number of test batches = ', len(test_batches))

    test_ranking(model, test_batches)

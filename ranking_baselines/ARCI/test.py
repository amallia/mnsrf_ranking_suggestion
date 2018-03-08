###############################################################################
# Author: Wasi Ahmad
# Project: ARC-I: Convolutional Matching Model
# Date Created: 7/18/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os, numpy
from model import CNN_ARC_I
from rank_metrics import mean_average_precision, NDCG, MRR

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    _map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        queries, docs, click_labels = helper.batch_to_tensor(test_batches[batch_no - 1], model.dictionary, model.config)
        if model.config.cuda:
            queries = queries.cuda()
            docs = docs.cuda()
            click_labels = click_labels.cuda()

        score = model(queries, docs)
        _map += mean_average_precision(score, click_labels)
        mrr += MRR(score, click_labels)
        ndcg_1 += NDCG(score, click_labels, 1)
        ndcg_3 += NDCG(score, click_labels, 3)
        ndcg_5 += NDCG(score, click_labels, 5)
        ndcg_10 += NDCG(score, click_labels, 10)

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
    # Load the saved pre-trained model
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = CNN_ARC_I(dictionary, embeddings_index, args)

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

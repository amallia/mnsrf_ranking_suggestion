###############################################################################
# Author: Wasi Ahmad
# Project: Sequence to sequence query generation
# Date Created: 06/04/2017
#
# File Description: This script evaluates query generation based on ranking 
# eval metrics.
###############################################################################

import torch, helper, util, os, numpy, data
from model import Seq2Seq
from torch.autograd import Variable
from rank_metrics import NDCG, MRR

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def suggest_next_query(model, current_query, q_len, next_query):
    # encode current query
    embedded_cq = model.embedding(current_query)
    encoded_cq = model.encoder(embedded_cq, q_len)

    if model.config.pool_type == 'max':
        hidden_states = torch.max(encoded_cq, 1)[0].squeeze()
    elif model.config.pool_type == 'mean':
        hidden_states = torch.sum(encoded_cq, 1).squeeze() / encoded_cq.size(1)
    elif model.config.pool_type == 'last':
        if model.num_directions == 2:
            hidden_states = torch.cat(
                (encoded_cq[:, -1, :model.config.nhid_enc], encoded_cq[:, -1, model.config.nhid_enc:]), 1)
        else:
            hidden_states = encoded_cq[:, -1, :]

    # Initialize hidden states of decoder with the last hidden states of the encoder
    if model.config.model is 'LSTM':
        cell_states = Variable(torch.zeros(*hidden_states.size()))
        if model.config.cuda:
            cell_states = cell_states.cuda()
        decoder_hidden = (hidden_states.unsqueeze(0).contiguous(), cell_states.unsqueeze(0).contiguous())
    else:
        decoder_hidden = hidden_states.unsqueeze(0).contiguous()

    if model.config.attn_type:
        decoder_context = Variable(torch.zeros(*hidden_states.size())).unsqueeze(1)
        if model.config.cuda:
            decoder_context = decoder_context.cuda()

    gen_prob = 0
    for di in range(next_query.size(1) - 1):
        embedded_decoder_input = model.embedding(next_query[:, di]).unsqueeze(1)
        if model.config.attn_type:
            decoder_output, decoder_hidden, decoder_context, attn_weights = model.decoder(embedded_decoder_input,
                                                                                          decoder_hidden,
                                                                                          decoder_context,
                                                                                          encoded_cq)
        else:
            decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)

        gen_prob += decoder_output.squeeze()[next_query[0, di + 1].data].data.cpu().numpy()[0]

    return gen_prob


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = Seq2Seq(dictionary, embeddings_index, args)
    print(model)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('model, embedding index and dictionary loaded.')
    model.eval()

    # load the test dataset
    test_corpus = data.Corpus(args.tokenize, args.max_query_length)
    test_corpus.parse(args.data + 'dev.txt', args.max_example, False)
    print('test set size = ', len(test_corpus.data))

    candidate_map = dict()
    with open('../data/anchor_candidates.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split(':::')
            candidate_map[tokens[0]] = []
            for i in range(1, len(tokens)):
                candidate_map[tokens[0]].append(tokens[i])

    score_batch, target = [], []
    mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0
    total_eval_batch = 0
    for current_query, next_query in test_corpus.data:
        current_query_text = ' '.join(current_query.query_terms[1:-1])
        if current_query_text not in candidate_map:
            continue
        cands = []
        for query_text in candidate_map[current_query_text]:
            query = data.Query()
            query.add_text(query_text, args.tokenize, args.max_query_length)
            cands.append(query)
        cands.append(next_query)

        scores = []
        for cand in cands:
            q1_var, q1_len, q2_var, q2_len = helper.batch_to_tensor([(current_query, cand)], dictionary)
            if model.config.cuda:
                q1_var = q1_var.cuda()  # batch_size x max_len
                q2_var = q2_var.cuda()  # batch_size x max_len

            score = suggest_next_query(model, q1_var, q1_len, q2_var)
            scores.append(score)

        score_batch.append(scores)
        target.append([0] * 20 + [1])
        if len(score_batch) == 256:
            batch_scores = Variable(torch.from_numpy(numpy.asarray(score_batch)))
            batch_target = Variable(torch.from_numpy(numpy.asarray(target)))
            if model.config.cuda:
                batch_scores = batch_scores.cuda()
                batch_target = batch_target.cuda()
            mrr += MRR(batch_scores, batch_target)
            ndcg_1 += NDCG(batch_scores, batch_target, 1)
            ndcg_3 += NDCG(batch_scores, batch_target, 3)
            ndcg_5 += NDCG(batch_scores, batch_target, 5)
            ndcg_10 += NDCG(batch_scores, batch_target, 10)
            total_eval_batch += 1
            score_batch, target = [], []

    print('MRR - ', mrr / total_eval_batch)
    print('NDCG@1 - ', ndcg_1 / total_eval_batch)
    print('NDCG@3 - ', ndcg_3 / total_eval_batch)
    print('NDCG@5 - ', ndcg_5 / total_eval_batch)
    print('NDCG@10 - ', ndcg_10 / total_eval_batch)


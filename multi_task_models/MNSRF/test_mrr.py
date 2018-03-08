###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Framework
# Date Created: 7/15/2017
#
# File Description: This script contains code for testing.
###############################################################################

import util, helper, os, torch, data, numpy
from torch.autograd import Variable
from rank_metrics import NDCG, MRR
from model import NSRF

args = util.get_args()


def suggest_next_query(model, session_queries, session_query_length, next_query_tensor):
    # query encoding
    embedded_queries = model.embedding(session_queries.view(-1, session_queries.size(-1)))
    encoded_queries = model.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
    encoded_queries = model.apply_pooling(encoded_queries, model.config.pool_type)

    # encoded_queries: batch_size x session_length x (nhid_query * self.num_directions)
    encoded_queries = encoded_queries.contiguous().view(*session_queries.size()[:-1], -1)

    # session level encoding
    sess_query_hidden = model.session_query_encoder.init_weights(encoded_queries.size(0))
    hidden_states = []

    # loop over all the queries in a session
    for idx in range(encoded_queries.size(1)):
        # update session-level query encoder state using query representations
        sess_q_out, sess_q_hidden = model.session_query_encoder(encoded_queries[:, idx, :].unsqueeze(1),
                                                                sess_query_hidden)
        hidden_states.append(sess_q_out.squeeze(1))

    hidden_states = torch.stack(hidden_states, 1)

    # only take the second last hidden state which stands for the anchor query in the session
    hidden_states = hidden_states[:, -2, :].unsqueeze(0)
    cell_states = Variable(torch.zeros(*hidden_states.size()))
    if model.config.cuda:
        hidden_states = hidden_states.cuda()
        cell_states = cell_states.cuda()

    # Initialize hidden states of decoder with the second last hidden states of the session encoder
    decoder_hidden = (hidden_states, cell_states)
    gen_prob = 0
    for di in range(next_query_tensor.size(1) - 1):
        embedded_decoder_input = model.embedding(next_query_tensor[:, di]).unsqueeze(1)
        decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)

        gen_prob += decoder_output.squeeze()[next_query_tensor[0, di + 1].data].data.cpu().numpy()[0]

    return gen_prob


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = NSRF(dictionary, embeddings_index, args)
    print(model)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('model, embedding index and dictionary loaded.')
    model.eval()

    # load the test dataset
    test_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example, load_query_only=True)
    print('test set size = ', len(test_corpus))

    candidate_map = dict()
    with open(args.data + 'anchor_candidates.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split(':::')
            candidate_map[tokens[0]] = []
            for i in range(1, len(tokens)):
                candidate_map[tokens[0]].append(tokens[i])

    score_batch, target = [], []
    mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0
    total_eval_batch = 0
    for sess_len, sessions in test_corpus.data.items():
        for session in sessions:
            anchor_query_text = ' '.join(session.queries[sess_len - 2].query_terms[1:-1])
            if anchor_query_text not in candidate_map:
                continue
            cands = []
            for query_text in candidate_map[anchor_query_text]:
                query = data.Query()
                query.add_text(query_text, args.tokenize, args.max_query_length)
                cands.append(query)
            cands.append(session.queries[sess_len - 1])

            scores = []
            session_queries, session_query_length = helper.session_queries_to_tensor([session], dictionary, iseval=True)
            for cand in cands:
                next_query_tensor = helper.sequence_to_tensors(cand.query_terms, len(cand.query_terms), dictionary)
                next_query_tensor = Variable(next_query_tensor).unsqueeze(0)
                if model.config.cuda:
                    session_queries = session_queries.cuda()  # 1 x session_length x max_query_length
                    session_query_length = session_query_length.cuda()  # 1 x session_length
                    next_query_tensor = next_query_tensor.cuda()  # 1 x max_query_length

                score = suggest_next_query(model, session_queries, session_query_length, next_query_tensor)
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

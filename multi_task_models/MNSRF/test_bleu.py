###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Framework
# Date Created: 7/29/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import os, util, helper, data, multi_bleu, torch
from torch.autograd import Variable
from data import Session
from model import NSRF

args = util.get_args()


def suggest_next_query(model, session_queries, session_query_length, dictionary):
    # query encoding
    embedded_queries = model.embedding(session_queries.view(-1, session_queries.size(-1)))
    encoded_queries = model.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
    encoded_queries = model.apply_pooling(encoded_queries, model.config.pool_type)
    # encoded_queries: batch_size x session_length x (nhid_query * self.num_directions)
    encoded_queries = encoded_queries.view(*session_queries.size()[:-1], -1)

    # session level encoding
    sess_q_hidden = model.session_query_encoder.init_weights(encoded_queries.size(0))
    hidden_states, cell_states = [], []
    # loop over all the queries in a session
    for idx in range(encoded_queries.size(1)):
        # update session-level query encoder state using query representations
        sess_q_out, sess_q_hidden = model.session_query_encoder(encoded_queries[:, idx, :].unsqueeze(1), sess_q_hidden)
        # -1: only consider hidden states of the last layer
        if model.config.model == 'LSTM':
            hidden_states.append(sess_q_hidden[0][-1])
            cell_states.append(sess_q_hidden[1][-1])
        else:
            hidden_states.append(sess_q_hidden[0][-1])

    hidden_states = torch.stack(hidden_states, 1)
    hidden_states = hidden_states[:, -1, :].contiguous().view(-1, hidden_states.size(-1)).unsqueeze(0)
    if model.config.model == 'LSTM':
        cell_states = torch.stack(cell_states, 1)
        cell_states = cell_states[:, -1, :].contiguous().view(-1, cell_states.size(-1)).unsqueeze(0)
        decoder_hidden = (hidden_states, cell_states)
    else:
        decoder_hidden = hidden_states

    sos_token_index = dictionary.word2idx['<s>']
    eos_token_index = dictionary.word2idx['</s>']

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    for di in range(model.config.max_query_length + 1):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        embedded_decoder_input = model.embedding(decoder_input).unsqueeze(1)
        decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            break
        else:
            decoded_words.append(dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    return " ".join(decoded_words)


def evaluate(model, dictionary, session_queries):
    session = Session()
    session.queries = session_queries
    session_queries, session_query_length, rel_docs, rel_docs_length, doc_labels = helper.session_to_tensor(
        [session], dictionary, iseval=True)
    if model.config.cuda:
        session_queries = session_queries.cuda()
        session_query_length = session_query_length.cuda()
    return suggest_next_query(model, session_queries, session_query_length, dictionary)


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = NSRF(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('model, embedding index and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example)
    print('test set size = ', len(test_corpus))

    targets, candidates = [], []
    fw = open(args.save_path + 'predictions_mmt.txt', 'w')
    for sess_len, sessions in test_corpus.data.items():
        for sess in sessions:
            for i in range(len(sess) - 1):
                target = evaluate(model, dictionary, sess.queries[:i + 1])
                candidate = " ".join(sess.queries[i + 1].query_terms[1:-1])
                targets.append(target)
                candidates.append(candidate)
                inp = []
                for query in sess.queries[:i + 1]:
                    inp.append(' '.join(query.query_terms[1:-1]))
                fw.write(', '.join(inp) + ' <:::> ' + candidate + ' <:::> ' + target + '\n')
    fw.close()

    print("target size = ", len(targets))
    print("candidate size = ", len(candidates))
    multi_bleu.print_multi_bleu(targets, candidates)

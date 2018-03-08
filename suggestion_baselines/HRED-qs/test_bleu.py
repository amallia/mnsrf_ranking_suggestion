###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import os, util, helper, data, multi_bleu, torch
from torch.autograd import Variable
from data import Session
from model import HRED_QS

args = util.get_args()


def suggest_next_query(model, dictionary, batch_session, session_query_length):
    # query encoding
    embedded_queries = model.embedding(batch_session.view(-1, batch_session.size(-1)))
    encoded_queries = model.query_encoder(embedded_queries, session_query_length.view(-1).data.cpu().numpy())
    encoded_queries = model.apply_pooling(encoded_queries, model.config.pool_type)

    # encoded_queries: batch_size x session_length x (nhid_query * self.num_directions)
    encoded_queries = encoded_queries.contiguous().view(*batch_session.size()[:-1], -1)

    # session level encoding
    sess_hidden = model.session_encoder.init_weights(batch_session.size(0))
    hidden_states, cell_states = [], []
    for idx in range(encoded_queries.size(1)):
        sess_output, sess_hidden = model.session_encoder(encoded_queries[:, idx, :].unsqueeze(1), sess_hidden)
        if model.config.model == 'LSTM':
            hidden_states.append(sess_hidden[0][-1])
            cell_states.append(sess_hidden[1][-1])
        else:
            hidden_states.append(sess_hidden[-1])

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
    sess = Session()
    sess.queries = session_queries
    session_tensor, query_lengths = helper.session_to_tensor([sess], dictionary, True)
    if args.cuda:
        session_tensor = session_tensor.cuda()
        query_lengths = query_lengths.cuda()
    return suggest_next_query(model, dictionary, session_tensor, query_lengths)


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = HRED_QS(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('model, embedding index and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.tokenize, args.max_query_length)
    test_corpus.parse(args.data + 'test.txt', args.max_example)
    print('test set size = ', len(test_corpus))

    targets, candidates = [], []
    fw = open(args.save_path + 'predictions_hredqs.txt', 'w')
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

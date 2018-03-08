###############################################################################
# Author: Wasi Ahmad
# Project: Sequence to sequence query generation
# Date Created: 06/04/2017
#
# File Description: This script visualizes query and document representations.
###############################################################################

import torch, helper, util, os, numpy, data, multi_bleu
from model import Seq2Seq
from torch.autograd import Variable

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def generate_next_query(model, query, query_len, dictionary):
    # encode the query
    embedded_q = model.embedding(query)
    encoded_q, hidden = model.encoder(embedded_q, query_len)

    if model.config.bidirection:
        h_t, c_t = hidden[0][-2:], hidden[1][-2:]
        decoder_hidden = torch.cat((h_t[0].unsqueeze(0), h_t[1].unsqueeze(0)), 2), \
                         torch.cat((c_t[0].unsqueeze(0), c_t[1].unsqueeze(0)), 2)
    else:
        h_t, c_t = hidden[0][-1], hidden[1][-1]
        decoder_hidden = h_t, c_t

    if model.config.attn_type:
        decoder_context = Variable(torch.zeros(encoded_q.size(0), encoded_q.size(2))).unsqueeze(1)
        if model.config.cuda:
            decoder_context = decoder_context.cuda()

    sos_token_index = dictionary.word2idx['<s>']
    eos_token_index = dictionary.word2idx['</s>']

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    for di in range(model.config.max_query_length + 1):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        embedded_decoder_input = model.embedding(decoder_input).unsqueeze(1)
        if model.config.attn_type:
            decoder_output, decoder_hidden, decoder_context, attn_weights = model.decoder(embedded_decoder_input,
                                                                                          decoder_hidden,
                                                                                          decoder_context,
                                                                                          encoded_q)
        else:
            decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            break
        else:
            decoded_words.append(dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    return " ".join(decoded_words)


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
    test_corpus.parse(args.data + 'test.txt', args.max_example, whole_session=True)
    print('test set size = ', len(test_corpus.data))

    targets, candidates = [], []
    if args.attn_type:
        fw = open(args.save_path + 'seq2seq_attn_predictions.txt', 'w')
    else:
        fw = open(args.save_path + 'seq2seq_predictions.txt', 'w')
    for prev_q, current_q in test_corpus.data:
        q1_var, q1_len, q2_var, q2_len = helper.batch_to_tensor([(prev_q, current_q)], dictionary,
                                                                reverse=args.reverse, iseval=True)
        if args.cuda:
            q1_var = q1_var.cuda()  # batch_size x max_len
            q2_var = q2_var.cuda()  # batch_size x max_len
            q2_len = q2_len.cuda()  # batch_size

        target = generate_next_query(model, q1_var, q1_len, dictionary)
        candidate = " ".join(current_q.query_terms[1:-1])
        targets.append(target)
        candidates.append(candidate)
        fw.write(candidate + '\t' + target + '\n')
    fw.close()

    print("target size = ", len(targets))
    print("candidate size = ", len(candidates))
    multi_bleu.print_multi_bleu(targets, candidates)

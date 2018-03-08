###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script handles preprocessing and creation of vocabulary.
###############################################################################

import os, operator, util, json, helper


class Vocabulary(object):
    def __init__(self, order_n_gram=5):
        self.order_n_gram = order_n_gram
        self.word2freq = {}

    def form_vocabulary(self, in_file, tokenize):
        """Creates the vocabulary."""
        assert os.path.exists(in_file)
        with open(in_file, 'r') as f:
            for line in f:
                session = json.loads(line)
                assert len(session['query']) == len(session['clicks'])
                for qidx in range(len(session['query'])):
                    query_terms = helper.tokenize(session['query'][qidx][0], tokenize)
                    query_letter_n_grams = self.get_letter_n_grams(query_terms, self.order_n_gram)
                    self.add_letter_n_grams(query_letter_n_grams)
                    for i in range(len(session['clicks'][qidx])):
                        doc_title = session['clicks'][qidx][i][1]
                        title_terms = helper.tokenize(doc_title, tokenize)
                        doc_letter_n_grams = self.get_letter_n_grams(title_terms, self.order_n_gram)
                        self.add_letter_n_grams(doc_letter_n_grams)

    def get_letter_n_grams(self, tokens, n):
        letter_n_grams = []
        for i in range(len(tokens)):
            if tokens[i] != '<unk>':
                for j in range(1, n + 1):
                    if j > len(tokens[i]):
                        break
                    else:
                        # create letter_n_grams where n = j
                        letter_n_grams.extend(self.find_letter_ngrams(tokens[i], j))

        return letter_n_grams

    @staticmethod
    def find_letter_ngrams(word, n):
        return [''.join(list(a)) for a in zip(*[word[i:] for i in range(n)])]

    def add_letter_n_grams(self, n_grams):
        for token in n_grams:
            if token not in self.word2freq:
                self.word2freq[token] = 1
            else:
                self.word2freq[token] += 1

    def contains(self, word):
        return True if word in self.word2freq else False

    def save_vocabulary(self, path, filename):
        assert os.path.exists(path)
        sorted_x = sorted(self.word2freq.items(), key=operator.itemgetter(1), reverse=True)
        with open(os.path.join(path, filename), 'w') as f:
            for word, freq in sorted_x:
                f.write(word + ',' + str(freq) + '\n')

    def __len__(self):
        return len(self.word2freq)


if __name__ == '__main__':
    args = util.get_args()
    # if output directory doesn't exist, create it
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    vocab = Vocabulary()
    vocab.form_vocabulary(args.data + 'train.txt', args.tokenize)
    print('dictionary size - ', len(vocab))
    vocab.save_vocabulary(args.save_path, 'vocab.csv')

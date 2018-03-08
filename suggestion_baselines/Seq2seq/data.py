###############################################################################
# Author: Wasi Ahmad
# Project: Sequence to sequence query generation
# Date Created: 06/04/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, helper, json
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = '<p>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1
        self.unk_token = '<unk>'
        self.idx2word.append(self.unk_token)
        self.word2idx[self.unk_token] = len(self.idx2word) - 1

    def build_dict(self, sample, max_words):
        word_count = Counter()
        for query1, query2 in sample.data:
            word_count.update(query1.query_terms)
            word_count.update(query2.query_terms)

        most_common = word_count.most_common(max_words)
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Query(object):
    def __init__(self):
        self.query_terms = []

    def add_text(self, text, tokenize, max_length):
        content_terms = helper.tokenize(text, tokenize)
        if len(content_terms) > max_length:
            self.query_terms = ['<s>'] + content_terms[:max_length] + ['</s>']
        else:
            self.query_terms = ['<s>'] + content_terms + ['</s>']


class Corpus(object):
    def __init__(self, _tokenize, max_q_len):
        self.tokenize = _tokenize
        self.data = []
        self.max_q_len = max_q_len

    def parse(self, in_file, max_example=None, whole_session=True):
        """Parses the content of a file."""
        assert os.path.exists(in_file)

        with open(in_file, 'r') as f:
            for line in f:
                session = json.loads(line)
                if whole_session:
                    prev_query = None
                    for qidx in range(len(session['query'])):
                        current_query = Query()
                        current_query.add_text(session['query'][qidx][0], self.tokenize, self.max_q_len)
                        if prev_query:
                            self.data.append((prev_query, current_query))
                        prev_query = current_query
                        if len(self.data) == max_example:
                            break
                else:
                    total_query_in_session = len(session['query'])
                    current_q, prev_q = Query(), Query()
                    current_q.add_text(session['query'][total_query_in_session - 1][0], self.tokenize, self.max_q_len)
                    prev_q.add_text(session['query'][total_query_in_session - 2][0], self.tokenize, self.max_q_len)
                    self.data.append((prev_q, current_q))
                    if len(self.data) == max_example:
                        break

###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
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
        for key, value in sample.data.items():
            for session in value:
                for query in session.queries:
                    word_count.update(query.query_terms)

        print('total unique word = ', len(word_count))
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


class Session(object):
    def __init__(self):
        self.queries = []

    def add_query(self, query):
        self.queries.append(query)

    def __len__(self):
        return len(self.queries)


class Corpus(object):
    def __init__(self, _tokenize, max_q_len):
        self.tokenize = _tokenize
        self.data = dict()
        self.max_q_len = max_q_len

    def parse(self, in_file, max_example=None):
        """Parses the content of a file."""
        assert os.path.exists(in_file)

        total_session = 0
        with open(in_file, 'r') as f:
            for line in f:
                session = json.loads(line)
                sess_obj = Session()
                for qidx in range(len(session['query'])):
                    query = Query()
                    query.add_text(session['query'][qidx][0], self.tokenize, self.max_q_len)
                    sess_obj.add_query(query)
                if len(sess_obj) in self.data:
                    self.data[len(sess_obj)].append(sess_obj)
                else:
                    self.data[len(sess_obj)] = [sess_obj]
                if total_session == max_example:
                    break

    def __len__(self):
        length = 0
        for key, value in self.data.items():
            length += len(value)
        return length

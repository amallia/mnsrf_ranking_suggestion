###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, helper, json
from collections import Counter
from itertools import chain


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def build_dict(self, sample, max_words):
        word_count = Counter()
        for query in sample.data:
            word_count.update(list(chain.from_iterable(query.letter_trigrams)))
            for doc in query.rel_docs:
                word_count.update(list(chain.from_iterable(doc.letter_trigrams)))

        most_common = word_count.most_common(max_words) if max_words > 0 else word_count.most_common()
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self):
        self.letter_trigrams = []
        self.is_clicked = False

    def add_content(self, text, tokenize, max_len):
        content_terms = helper.tokenize(text, tokenize)
        content_terms = content_terms if len(content_terms) <= max_len else content_terms[:max_len]
        content_terms = ['<s>'] + content_terms + ['<s>']
        content_terms = ['#' + item + '#' for item in content_terms]
        for i in range(len(content_terms)):
            # create letter-trigrams
            word = content_terms[i]
            letter_trigrams_for_words = []
            for j in range(0, len(word) - 2):
                letter_trigrams_for_words.append(word[j:j + 3])
            self.letter_trigrams.append(letter_trigrams_for_words)

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self):
        self.letter_trigrams = []
        self.rel_docs = []

    def add_text(self, text, tokenize, max_len):
        content_terms = helper.tokenize(text, tokenize)
        content_terms = content_terms if len(content_terms) <= max_len else content_terms[:max_len]
        content_terms = ['<s>'] + content_terms + ['<s>']
        content_terms = ['#' + item + '#' for item in content_terms]
        for i in range(len(content_terms)):
            # create letter-trigrams
            word = content_terms[i]
            letter_trigrams_for_words = []
            for j in range(0, len(word) - 2):
                letter_trigrams_for_words.append(word[j:j + 3])
            self.letter_trigrams.append(letter_trigrams_for_words)

    def add_document(self, document):
        self.rel_docs.append(document)


class Corpus(object):
    def __init__(self, _tokenize, max_q_length, max_d_length):
        self.tokenize = _tokenize
        self.max_query_len = max_q_length
        self.max_doc_len = max_d_length
        self.data = []

    def parse(self, in_file, max_example):
        """Parses the content of a file."""
        assert os.path.exists(in_file)

        with open(in_file, 'r') as f:
            for line in f:
                session = json.loads(line)
                assert len(session['query']) == len(session['clicks'])
                for qidx in range(len(session['query'])):
                    query = Query()
                    query.add_text(session['query'][qidx][0], self.tokenize, self.max_query_len)
                    for i in range(len(session['clicks'][qidx])):
                        doc = Document()
                        doc_title = session['clicks'][qidx][i][1]
                        doc_label = session['clicks'][qidx][i][2]
                        doc.add_content(doc_title, self.tokenize, self.max_doc_len)
                        if int(doc_label) == 1:
                            doc.set_clicked()
                        query.add_document(doc)
                    self.data.append(query)
                    if len(self.data) == max_example:
                        return

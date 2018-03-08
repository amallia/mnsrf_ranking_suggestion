###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, json, helper


class Dictionary(object):
    def __init__(self, order_n_gram):
        self.order_n_gram = order_n_gram
        self.word2idx = {}
        self.idx2word = []

    def load_dictionary(self, path, filename, top_n_words):
        with open(os.path.join(path, filename), 'r') as f:
            counter = 0
            for line in f:
                token = line.strip().split(',')[0]
                self.idx2word.append(token)
                self.word2idx[token] = len(self.idx2word) - 1
                counter += 1
                if counter == top_n_words:
                    break

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self, order_n_gram):
        self.order_n_gram = order_n_gram
        self.letter_n_grams = []
        self.is_clicked = False

    @staticmethod
    def find_letter_ngrams(word, n):
        return [''.join(list(a)) for a in zip(*[word[i:] for i in range(n)])]

    def add_content(self, text, tokenize, max_len):
        content_terms = helper.tokenize(text, tokenize)
        content_terms = content_terms if len(content_terms) <= max_len else content_terms[:max_len]
        for i in range(len(content_terms)):
            # create letter-trigrams
            word = content_terms[i]
            word_letter_n_grams = []
            for j in range(1, self.order_n_gram + 1):
                if j > len(word):
                    break
                else:
                    # create letter_n_grams where n = j
                    word_letter_n_grams.extend(self.find_letter_ngrams(word, j))
            if word_letter_n_grams:
                self.letter_n_grams.append(word_letter_n_grams)

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self, order_n_gram):
        self.order_n_gram = order_n_gram
        self.letter_n_grams = []
        self.rel_docs = []

    @staticmethod
    def find_letter_ngrams(word, n):
        return [''.join(list(a)) for a in zip(*[word[i:] for i in range(n)])]

    def add_text(self, text, tokenize, max_len):
        content_terms = helper.tokenize(text, tokenize)
        content_terms = content_terms if len(content_terms) <= max_len else content_terms[:max_len]
        for i in range(len(content_terms)):
            # create letter-trigrams
            word = content_terms[i]
            word_letter_n_grams = []
            for j in range(1, self.order_n_gram + 1):
                if j > len(word):
                    break
                else:
                    # create letter_n_grams where n = j
                    word_letter_n_grams.extend(self.find_letter_ngrams(word, j))
            if word_letter_n_grams:
                self.letter_n_grams.append(word_letter_n_grams)

    def add_document(self, document):
        self.rel_docs.append(document)


class Corpus(object):
    def __init__(self, _tokenize, max_q_length, max_d_length, order_n_gram=5):
        self.order_n_gram = order_n_gram
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
                    query = Query(self.order_n_gram)
                    query.add_text(session['query'][qidx][0], self.tokenize, self.max_query_len)
                    for i in range(len(session['clicks'][qidx])):
                        doc = Document(self.order_n_gram)
                        doc_title = session['clicks'][qidx][i][1]
                        doc_label = session['clicks'][qidx][i][2]
                        doc.add_content(doc_title, self.tokenize, self.max_doc_len)
                        if int(doc_label) == 1:
                            doc.set_clicked()
                        query.add_document(doc)
                    self.data.append(query)
                    if len(self.data) == max_example:
                        return

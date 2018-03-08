###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Framework
# Date Created: 7/15/2017
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
                    for doc in query.rel_docs:
                        word_count.update(doc.body)

        print('total unique word = ', len(word_count))
        most_common = word_count.most_common(max_words)
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self):
        self.body = []
        self.is_clicked = False

    def add_content(self, text, tokenize, max_length):
        content_terms = helper.tokenize(text, tokenize)
        if len(content_terms) > max_length:
            self.body = ['<s>'] + content_terms[:max_length] + ['</s>']
        else:
            self.body = ['<s>'] + content_terms + ['</s>']

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self):
        self.query_terms = []
        self.rel_docs = []

    def add_text(self, text, tokenize, max_length):
        content_terms = helper.tokenize(text, tokenize)
        if len(content_terms) > max_length:
            self.query_terms = ['<s>'] + content_terms[:max_length] + ['</s>']
        else:
            self.query_terms = ['<s>'] + content_terms + ['</s>']

    def add_document(self, document):
        self.rel_docs.append(document)


class Session(object):
    def __init__(self):
        self.queries = []

    def add_query(self, query):
        self.queries.append(query)

    def __len__(self):
        return len(self.queries)


class Corpus(object):
    def __init__(self, _tokenize, max_q_len, max_doc_len):
        self.tokenize = _tokenize
        self.data = dict()
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len

    def parse(self, in_file, max_example=None, load_query_only=False):
        """Parses the content of a file."""
        assert os.path.exists(in_file)

        total_session = 0
        with open(in_file, 'r') as f:
            for line in f:
                session = json.loads(line)
                assert len(session['query']) == len(session['clicks'])

                sess_obj = Session()
                for qidx in range(len(session['query'])):
                    query = Query()
                    query.add_text(session['query'][qidx][0], self.tokenize, self.max_q_len)
                    if not load_query_only:
                        for i in range(len(session['clicks'][qidx])):
                            doc = Document()
                            doc_body = session['clicks'][qidx][i][1]
                            doc_label = session['clicks'][qidx][i][2]
                            doc.add_content(doc_body, self.tokenize, self.max_doc_len)
                            if int(doc_label) == 1:
                                doc.set_clicked()
                            query.add_document(doc)
                    sess_obj.add_query(query)
                if len(sess_obj) in self.data:
                    self.data[len(sess_obj)].append(sess_obj)
                else:
                    self.data[len(sess_obj)] = [sess_obj]
                total_session += 1
                if total_session == max_example:
                    break

    def __len__(self):
        length = 0
        for key, value in self.data.items():
            length += len(value)
        return length

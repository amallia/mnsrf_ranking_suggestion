###############################################################################
# Author: Wasi Ahmad
# Project: Embedding Space Model
# Date Created: 7/18/2017
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Embedding Space Model')
    parser.add_argument('--data', type=str, default='../data/aol/v1/',
                        help='location of the data corpus')
    parser.add_argument('--max_words', type=int, default=80000,
                        help='maximum number of words (top ones) to be added to dictionary)')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--max_doc_length', type=int, default=20,
                        help='maximum length of a document')
    parser.add_argument('--max_query_length', type=int, default=10,
                        help='maximum length of a query')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding version')
    parser.add_argument('--word_vectors_directory', type=str, default='../data/glove/',
                        help='Path of GloVe word embeddings')

    args = parser.parse_args()
    return args

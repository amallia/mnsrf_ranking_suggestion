###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Match Tensor')
    parser.add_argument('--data', type=str, default='../../data/aol/v1/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--max_words', type=int, default=80000,
                        help='maximum number of words (top ones) to be added to dictionary)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--emtraining', action='store_true',
                        help='train embedding layer')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirectional recurrent unit')
    parser.add_argument('--featsize', type=int, default=40,
                        help='number of hidden units for first linear projection layer')
    parser.add_argument('--nhid_query', type=int, default=15,
                        help='number of hidden units per layer for query encoder')
    parser.add_argument('--nhid_doc', type=int, default=70,
                        help='number of hidden units per layer for document encoder')
    parser.add_argument('--nchannels', type=int, default=50,
                        help='number of match channels')
    parser.add_argument('--nfilters', type=int, default=6,
                        help='number of filters for convolution')
    parser.add_argument('--match_filter_size', type=int, default=20,
                        help='size of the match filter')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size')
    parser.add_argument('--early_stop', type=int, default=3,
                        help='early stopping criterion')
    parser.add_argument('--max_norm', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.20,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--max_doc_length', type=int, default=20,
                        help='maximum length of a document')
    parser.add_argument('--max_query_length', type=int, default=10,
                        help='maximum length of a query')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--print_every', type=int, default=500,
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=500,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--save_path', type=str, default='../mt_output/',
                        help='path to save the final model')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding version')
    parser.add_argument('--word_vectors_directory', type=str, default='../../data/glove/',
                        help='Path of GloVe word embeddings')

    args = parser.parse_args()
    return args

###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='DUET model')
    parser.add_argument('--data', type=str, default='../../data/aol/v1/',
                        help='location of the data corpus')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--max_words', type=int, default=-1,
                        help='maximum number of words (top ones) to be added to dictionary)')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--nfilters', type=int, default=300,
                        help='number of filters for convolution')
    parser.add_argument('--local_filter_size', type=int, default=1,
                        help='filter size for convolution in local model')
    parser.add_argument('--dist_filter_size', type=int, default=3,
                        help='filter size for convolution in distributed model')
    parser.add_argument('--pool_size', type=int, default=5,
                        help='max-pooling size second convolution on document in distributed model')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units in the hidden layers')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.20,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--max_doc_length', type=int, default=20,
                        help='maximum allowed length of a document')
    parser.add_argument('--max_query_length', type=int, default=10,
                        help='maximum allowed length of a query')
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
    parser.add_argument('--save_path', type=str, default='../duet_output/',
                        help='path to save the final model')

    args = parser.parse_args()
    return args

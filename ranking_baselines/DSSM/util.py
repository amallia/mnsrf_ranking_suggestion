###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
# Date Created: 7/18/2017
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Deep Semantic Similarity Model')
    parser.add_argument('--data', type=str, default='../../data/aol/v1/',
                        help='location of the data corpus')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--max_words', type=int, default=-1,
                        help='maximum number of words (top ones) to be added to dictionary)')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--nhid_output', type=int, default=128,
                        help='number of hidden units in the output layer')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units in the hidden layers')
    parser.add_argument('--lr', type=float, default=.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size')
    parser.add_argument('--early_stop', type=int, default=3,
                        help='early stopping criterion')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--print_every', type=int, default=200,
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=200,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--save_path', type=str, default='../dssm_output/',
                        help='path to save the final model')

    args = parser.parse_args()
    return args

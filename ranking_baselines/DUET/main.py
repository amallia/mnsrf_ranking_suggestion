###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, data, train, os, numpy, helper, sys
import torch
from torch import optim
from model import DUET

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# load train and dev dataset
train_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
train_corpus.parse(args.data + 'train.txt', args.max_example)
print('train set size = ', len(train_corpus.data))
dev_corpus = data.Corpus(args.tokenize, args.max_query_length, args.max_doc_length)
dev_corpus.parse(args.data + 'dev.txt', args.max_example)
print('development set size = ', len(dev_corpus.data))

dictionary = data.Dictionary(5)
dictionary.load_dictionary(args.save_path, 'vocab.csv', 5000)
print('vocabulary size = ', len(dictionary))

###############################################################################
# Build the model
###############################################################################

model = DUET(dictionary, args)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
best_loss = sys.maxsize

param_dict = helper.count_parameters(model)
print('#parameters = ', numpy.sum(list(param_dict.values())))

if args.cuda:
    model = model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = helper.load_from_checkpoint(args.resume, args.cuda)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

###############################################################################
# Train the model
###############################################################################

train = train.Train(model, optimizer, dictionary, args, best_loss)
train.train_epochs(train_corpus, dev_corpus, args.start_epoch, args.epochs)

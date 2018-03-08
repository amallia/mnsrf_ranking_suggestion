###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, helper, data, train, os, numpy, torch, sys
from torch import optim
from model import MatchTensor

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

dictionary = data.Dictionary()
dictionary.build_dict(train_corpus, args.max_words)
# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')
print('vocabulary size = ', len(dictionary))

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file, dictionary.word2idx)
print('number of OOV words = ', len(dictionary) - len(embeddings_index))

# ###############################################################################
# # Build the model
# ###############################################################################

model = MatchTensor(dictionary, embeddings_index, args)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
best_loss = sys.maxsize

param_dict = helper.count_parameters(model)
print('number of trainable parameters = ', numpy.sum(list(param_dict.values())))

# for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if len(cuda_visible_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
if args.cuda:
    model = model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# ###############################################################################
# # Train the model
# ###############################################################################

train = train.Train(model, optimizer, dictionary, embeddings_index, args, best_loss)
train.train_epochs(train_corpus, dev_corpus, args.start_epoch, args.epochs)

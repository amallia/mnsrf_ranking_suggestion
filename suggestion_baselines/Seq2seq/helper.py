###############################################################################
# Author: Wasi Ahmad
# Project: Sequence to sequence query generation
# Date Created: 06/04/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import re, os, glob, pickle, string, math, time, torch, util
import numpy as np
from nltk import word_tokenize
from collections import OrderedDict
from torch.autograd import Variable
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index):
    f = open(os.path.join(directory, file), 'w')
    for word, vec in embeddings_index.items():
        f.write(word + ' ' + ' '.join(str(x) for x in vec) + '\n')
    f.close()


def load_checkpoint(filename, from_gpu=True):
    """Load a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        return torch.load(filename)
    else:
        return torch.load(filename, map_location=lambda storage, loc: storage)


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def load_model_states_from_checkpoint(model, filename, tag, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint[tag])


def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())
    return param_dict


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return word_tokenize(s)
    else:
        return s.split()


def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def mask(sequence_length, seq_idx):
    batch_size = sequence_length.size(0)
    seq_range = torch.LongTensor([seq_idx])
    seq_range_expand = seq_range.expand(batch_size)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    return seq_range_expand < sequence_length


def batchify(data, bsz):
    """Transform data into batches."""
    np.random.shuffle(data)
    batched_data = []
    for i in range(len(data)):
        if i % bsz == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    return batched_data


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag)
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def show_plot(points):
    """Generates plots"""
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def sequence_to_tensors(sequence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sequence)):
        if dictionary.contains(sequence[i]):
            sen_rep[i] = dictionary.word2idx[sequence[i]]
        else:
            sen_rep[i] = dictionary.word2idx[dictionary.unk_token]

    return sen_rep


def batch_to_tensor(batch, dictionary, reverse=False, iseval=False):
    max_q1_len, max_q2_len = 0, 0
    for q1, q2 in batch:
        if max_q1_len < len(q1.query_terms):
            max_q1_len = len(q1.query_terms)
        if max_q2_len < len(q2.query_terms):
            max_q2_len = len(q2.query_terms)

    q1_tensor = torch.LongTensor(len(batch), max_q1_len)
    q1_length = np.zeros(len(batch), dtype=np.int)
    q2_tensor = torch.LongTensor(len(batch), max_q2_len)
    q2_length = torch.LongTensor(len(batch))
    for i in range(len(batch)):
        q1, q2 = batch[i]
        q1_length[i] = len(q1.query_terms)
        q2_length[i] = len(q2.query_terms)
        q1_terms = q1.query_terms[::-1] if reverse else q1.query_terms
        q1_tensor[i] = sequence_to_tensors(q1_terms, max_q1_len, dictionary)
        q2_tensor[i] = sequence_to_tensors(q2.query_terms, max_q2_len, dictionary)

    if iseval:
        return Variable(q1_tensor, volatile=True), q1_length, Variable(q2_tensor, volatile=True), \
               Variable(q2_length, volatile=True)
    else:
        return Variable(q1_tensor), q1_length, Variable(q2_tensor), Variable(q2_length)

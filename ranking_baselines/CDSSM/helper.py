###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import os, glob, pickle, math, time, torch, numpy
from nltk import wordpunct_tokenize, word_tokenize
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
from torch.autograd import Variable


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def load_from_checkpoint(filename, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    return checkpoint


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
            param_dict[name] = numpy.prod(param.size())
    return param_dict


def tokenize_and_normalize(s):
    """Tokenize and normalize string."""
    token_list = []
    tokens = wordpunct_tokenize(s.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


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
        return numpy.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return numpy.zeros(shape=dimension)


def batchify(data, bsz):
    """Transform data into batches."""
    numpy.random.shuffle(data)
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
    path_prefix = os.path.join(filepath, filetag + '_loss_plot_')
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


def sequence_to_tensors(sequence, max_len, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    seq_rep = numpy.zeros(shape=(max_len, len(dictionary)))
    for i in range(len(sequence)):
        for j in range(len(sequence[i])):
            if sequence[i][j] in dictionary.word2idx:
                seq_rep[i, dictionary.word2idx[sequence[i][j]]] += 1
    return seq_rep


def batch_to_tensor(batch, dictionary):
    max_q_len, max_d_len = 0, 0
    for i in range(len(batch)):
        max_q_len = max_q_len if max_q_len > len(batch[i].letter_trigrams) else len(batch[i].letter_trigrams)
        for j in range(len(batch[i].rel_docs)):
            max_d_len = max_d_len if max_d_len > len(batch[i].rel_docs[j].letter_trigrams) else len(
                batch[i].rel_docs[j].letter_trigrams)

    query_tensor = numpy.ndarray(shape=(len(batch), max_q_len, len(dictionary)))
    query_clicks = numpy.ndarray(shape=(len(batch), len(batch[0].rel_docs), max_d_len, len(dictionary)))
    click_labels = numpy.ndarray(shape=(len(batch), len(batch[0].rel_docs)))
    for i in range(len(batch)):
        query_tensor[i] = sequence_to_tensors(batch[i].letter_trigrams, max_q_len, dictionary)
        # shuffle the list of relevant documents
        numpy.random.shuffle(batch[i].rel_docs)
        for j in range(len(batch[i].rel_docs)):
            query_clicks[i, j] = sequence_to_tensors(batch[i].rel_docs[j].letter_trigrams, max_d_len, dictionary)
            click_labels[i, j] = 1 if batch[i].rel_docs[j].is_clicked else 0

    query_tensor = torch.from_numpy(query_tensor).type(torch.FloatTensor)
    query_clicks = torch.from_numpy(query_clicks).type(torch.FloatTensor)
    click_labels = torch.from_numpy(click_labels).type(torch.FloatTensor)
    return Variable(query_tensor), Variable(query_clicks), Variable(click_labels)

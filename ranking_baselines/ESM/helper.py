###############################################################################
# Author: Wasi Ahmad
# Project: Embedding Space Model
# Date Created: 7/18/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import os
import numpy as np
from nltk import word_tokenize


def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return word_tokenize(s)
    else:
        return s.split()

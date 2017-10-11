import random
from math import sqrt
import numpy as np

def cosine_similarity(v1, v2):
    '''Compute cosine similarity between two numpy vectors'''
    if len(v1) != len(v2):
        raise ValueError("Vectors are not the same length")
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

def load_space(infile):
    '''Read model file'''
    print "Reading model file",infile,"..."
    word_to_i = {}
    i_to_word = {}

    with open(infile) as f:
        dmlines=f.readlines()
        f.close()

    '''Make numpy array'''
    i = 0
    first_line = dmlines[0].rstrip('\n').split('\t')
    word_to_i[first_line[0]] = i
    i_to_word[i] = first_line[0]
    i+=1
    dm_mat = np.array([float(c) for c in first_line[1:]])
    dm_mat = dm_mat.reshape(1,len(dm_mat))
    for l in dmlines[1:]:
        items=l.rstrip('\n').split('\t')
        row=items[0]
        vec=np.array([float(c) for c in items[1:]])
        dm_mat = np.vstack([dm_mat,vec])
        word_to_i[items[0]] = i
        i_to_word[i] = items[0]
        i+=1  
    print "Shape of matrix:",dm_mat.shape
    return dm_mat, word_to_i, i_to_word


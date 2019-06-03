from gensim.models import KeyedVectors
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import heapq
import math



def load_word2vec_model(path):

    matrix = KeyedVectors.load_word2vec_format(path, binary=True)
    model = ('w2v', matrix)
    return model




def load_hyperwords_vocab(path):

    with open(path) as infile:
        vocab = [line.strip() for line in infile if len(line) > 0]

    wi_dict = dict([(w, i) for i, w in enumerate(vocab)])

    return wi_dict, vocab



def load_hyperwords_model(path, mtype):

    if mtype == 'sgns':
        matrix = np.load(path+'.words.npy')
        wi_dict, vocab = load_hyperwords_vocab(path+'.words.vocab')

    elif mtype == 'svd':
        matrix = np.load(path+'.ut.npy')
        wi_dict, vocab = load_hyperwords_vocab(path+'.words.vocab')
        # I don't know why, but they transpose the matrix in the hyperwords code:
        matrix = matrix.T

    elif mtype == 'ppmi':

        # code from hyperowrds hyperwords/representations/matrix_serializer.py
        loader = np.load(path+'.npz')
        matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        wi_dict, vocab = load_hyperwords_vocab(path+'.words.vocab')

    model = ('hyper', matrix, wi_dict, vocab)

    return model


def load_model(path, mtype):

    if mtype == 'w2v':

        model = load_word2vec_model(path)

    elif mtype != 'w2v':

        model = load_hyperwords_model(path, mtype)

    return model


def represent_word2vec(model, word):

    model_type, matrix = model

    vec = matrix[word]
    return vec



def represent_hyperwords(model, word):

    model_type, matrix, wi_dict, vocab = model

    if word in vocab:
        vec = matrix[wi_dict[word]]

    return vec



def represent(model, word):

    # Right now: no normalization. We might want to experiment with it though.

    model_type = model[0]


    if model_type =='w2v':
        matrix = model[1]
        if word in matrix.vocab:
            vec = represent_word2vec(model, word)
        else:
            vec = 'OOV'

    elif model_type == 'hyper':
        vocab = model[-1]
        if word in vocab:
            vec = represent_hyperwords(model, word)
        else:
            vec = 'OOV'
    else:
        vec = 'OOV'


    return vec

def normalize_matrix(matrix):

    # necessary for getting nearest neighbors

    # normalization for np matrix (sgns and svd)
    if type(matrix) == np.ndarray:

        norm = np.sqrt(np.sum(matrix * matrix, axis=1))
        matrix_norm = matrix / norm[:, np.newaxis]

    # normalization for sparse matrix
    else:
        m2 = matrix.copy()
        m2.data **= 2
        norm = np.reciprocal(np.sqrt(np.array(m2.sum(axis=1))[:, 0]))
        normalizer = dok_matrix((len(norm), len(norm)))
        normalizer.setdiag(norm)
        matrix_norm = normalizer.tocsr().dot(matrix)

    return matrix_norm
# Nearest neighbors

def normalize_vector(vec):

    """
    Calculate unit unit_vector
    Input: vector (resulting from a calculation)
    """
    if type(vec) == np.ndarray:
        mag = math.sqrt(sum([pow(value, 2) for value in vec]))

        unit_vec = []

        for value in vec:
            unit_vec.append(value/mag)
        unit_vec = np.array(unit_vec)

    else:

        v2 = vec.copy()
        v2  =  v2.power(2)
        norm = np.reciprocal(np.sqrt(np.array(v2.sum(axis=1))[:, 0]))
        normalizer = dok_matrix((len(norm), len(norm)))
        normalizer.setdiag(norm)
        unit_vec = normalizer.tocsr().dot(vec)


    return unit_vec



def get_nearest_neighbors(model, vec, n):

    model_type = model[0]
    matrix = model[1]

    ## Vector must always be normalized (results from a mean of
    #  not normalized vecs)

    vec_norm = normalize_vector(vec)

    if model_type == 'w2v':

        # Word2vec code automatically normalized vectors in model

        most_similar = matrix.similar_by_vector(vec_norm, topn = n)
        most_similar = [(cos, word) for word, cos in most_similar]


    elif model_type == 'hyper':

        wi, iw = model[2:]

        # normalize vectors:
        matrix_norm = normalize_matrix(matrix)


        if type(matrix) == np.ndarray:

            scores = matrix_norm.dot(vec_norm)
            most_similar = heapq.nlargest(n, zip(scores, iw))

        elif type(matrix) != np.ndarray:

            scores = matrix_norm.dot(vec_norm.T).T.tocsr()
            most_similar = heapq.nlargest(n, zip(scores.data, [iw[i] for i in scores.indices]))

    return most_similar


def get_cosine(vec1, vec2):

    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)

    cos = np.dot(vec1_norm, vec2_norm)

    return cos


def find_words_distance(model, cos, n):

    mtype, matrix, wi_dict, vocab = model

    vocab_clean = [word for word in vocab if word.isalpha()]

    pairs = []
    for word1 in vocab_clean:
        for word2 in vocab_clean:
            vec1 = represent(model, word1)
            vec2 = represent(model, word2)
            cos_pair = get_cosine(vec1, vec2)
            if round(cos_pair, 2) == round(cos, 2):
                pairs.append((word1, word2))
                print(word1, word2)
            if len(pairs) == n:
                break
        if len(pairs) == n:
            break
    print(pairs)

def test_cosine_pairs():
    word1 = 'star'
    word2 = 'talent'

    word3 = 'shine'
    word4 = 'eminent'
    word5 = 'beam'
    n = 5

    vec1 = represent(model, word1)
    vec2 = represent(model, word2)

    vec3 = represent(model, word3)
    vec4 = represent(model, word4)
    vec5 = represent(model, word5)

    cos1 = get_cosine(vec1, vec2)
    cos2 = get_cosine(vec4, vec5)
    cos3 = get_cosine(vec3, vec4)
    cos4 = get_cosine(vec3, vec5)

    print(f'Similarity of {word1} and {word2}: {cos1}')
    print(f'Similarity of {word4} and {word5}: {cos2}')
    print(f'Similarity of {word3} and {word4}: {cos3}')
    print(f'Similarity of {word3} and {word5}: {cos4}')

    n = 5
    find_words_distance(model, cos1, n)

def test_model(path, name):

    print(f'-----------{name}-------------')
    model = load_model(path, 'sgns')

    word1 = 'word'
    word2 = 'sentence'
    n = 10
    vec = represent(model, word1)
    vec2 = represent(model, word2)
    cos = get_cosine(vec, vec2)
    print(f'similarity between {word1} and {word2}: {cos}')
    print()
    most_similar = get_nearest_neighbors(model, vec, n)
    for cos, word in most_similar:
        print(word, cos)
    print('-----------------------------')
    print()






def main():

    #path_wiki_hyp_sgns = '/Users/piasommerauer/Data/dsm/wikipedia_full/sgns_pinit1/sgns_pinit1/sgns_rand_pinit1'

    #path_w2v = '/Users/piasommerauer/Data/dsm/word2vec/movies.bin'

    #path_w2v = '/Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'

    #path_hyp_sgns = '/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/sgns_rand_pinit1/sgns_rand_pinit1'

    #path_hyp_svd = '/Users/piasommerauer/Data/dsm/my_models/models/svd/svd'

    #path_hyp_ppmi = '/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/pmi/pmi'

    path_coha_1900_1 = '/Users/piasommerauer/Data/dsm/coha/coha_1900/sgns_rand_pinit1/sgns_rand_pinit1'
    path_coha_1900_2 = '/Users/piasommerauer/Data/dsm/coha/coha_1900/sgns_rand_pinit2/sgns_rand_pinit2'
    path_coha_1900_3 = '/Users/piasommerauer/Data/dsm/coha/coha_1900/sgns_rand_pinit3/sgns_rand_pinit3'


    #model = load_model(path_w2v, 'w2v')
    #model = load_model(path_hyp_sgns, 'sgns')
    #model = load_model(path_hyp_svd, 'svd')
    #model = load_model(path_wiki_hyp_sgns, 'sgns')

    test_model(path_coha_1900_1, 'pinit1')
    test_model(path_coha_1900_2, 'pinit2')
    test_model(path_coha_1900_3, 'pinit3')





if __name__ == '__main__':

    main()

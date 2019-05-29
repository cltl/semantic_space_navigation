import sys
import os
import numpy as np
from sklearn import preprocessing
from representations.sequentialembedding import SequentialEmbedding
from representations.sequentialembedding import SequentialSVDEmbedding
from representations.explicit import PositiveExplicit
import _pickle as pickle
import math

def get_sim_neighbors(corpus, target_word1, target_word2, year1, year2, n, results_dir):


    """Two options: either 2 differnt years and 1 target word
    or the same year and 2 target words"""

    if not os.path.isdir(results_dir+'neighbors'):

        os.mkdir(results_dir+'neighbors')

    results_words = 'neighbors/'+target_word1+'-'+target_word2+'-'+str(year1)+'-'+str(year2)+'.tsv'


    if (year1 != year2) and (target_word1 == target_word2):
        "using the same-neighbors measure"
        results_cosine = 'cosines-'+target_word1+'-n-'+str(n)+'.tsv'
        embedds = SequentialEmbedding.load(corpus, range(year1, year2+10, 10))
        embedd_year1 = embedds.get_embed(year1)
        embedd_year2 = embedds.get_embed(year2)

        neighbors_year1 = get_nearest_neighbors(embedd_year1, target_word1, n)
        neighbors_year2 = get_nearest_neighbors(embedd_year2, target_word1, n)

        union = get_union(neighbors_year1, neighbors_year2)

        filtered_union = filter_union(union, embedd_year1, embedd_year2, target_word1)
        vec1 = get_second_order_vector(embedd_year1, filtered_union, target_word1)
        vec2 = get_second_order_vector(embedd_year2, filtered_union, target_word1)

        neighbor_words1 = get_nearest_neighbor_words(neighbors_year1)
        neighbor_words2 = get_nearest_neighbor_words(neighbors_year2)



        neighbor_words1 = get_nearest_neighbor_words(neighbors_year1)
        neighbor_words2 = get_nearest_neighbor_words(neighbors_year2)

    elif (year1 == year2) and (target_word1 != target_word2):
        "using the pair nieghbors measure"
        results_cosine = 'cosines-'+target_word1+'-'+target_word2+'-n-'+str(n)+'.tsv'

        embedds = SequentialEmbedding.load(corpus, range(year1, year2+10, 10))
        embedd_year = embedds.get_embed(year1)


        neighbors_word1 = get_nearest_neighbors(embedd_year, target_word1, n)
        neighbors_word2 = get_nearest_neighbors(embedd_year, target_word2, n)

        union = get_union(neighbors_word1, neighbors_word2)

        vec1 = get_second_order_vector(embedd_year, union, target_word1)
        vec2 = get_second_order_vector(embedd_year, union, target_word2)

        neighbor_words1 = get_nearest_neighbor_words(neighbors_word1)
        neighbor_words2 = get_nearest_neighbor_words(neighbors_word2)

    cos = get_cosine(vec1, vec2)

    if os.path.isfile(results_dir+results_cosine):
        print('file exists')
        with open(results_dir+results_cosine) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_words, 'w') as outfile1:
        for word1, word2 in zip(neighbor_words1, neighbor_words2):
            #outfile1.write(word1.encode('ascii', 'ignore')+'\t'+word2.encode('ascii', 'ignore')+'\n')
            outfile1.write(word1+'\t'+word2+'\n')

    with open(results_dir+'/'+results_cosine, 'a') as outfile2:
        result = target_word1+'-'+target_word2+'\t'+str(year1)+'-'+str(year2)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile2.write(result)
    print(cos)

def get_sim_neighbors_svd(corpus, target_word1, target_word2, year1, year2, n, results_dir):


    """Two options: either 2 differnt years and 1 target word
    or the same year and 2 target words"""

    if not os.path.isdir(results_dir+'neighbors'):

        os.mkdir(results_dir+'neighbors')

    results_words = 'neighbors/'+target_word1+'-'+target_word2+'-'+str(year1)+'-'+str(year2)+'.tsv'

    if (year1 != year2) and (target_word1 == target_word2):
        results_cosine = 'cosines-'+target_word1+'-n-'+str(n)+'.tsv'
        embedds = SequentialSVDEmbedding.load(corpus, range(year1, year2+10, 10))
        embedd_year1 = embedds.get_embed(year1)
        embedd_year2 = embedds.get_embed(year2)

        neighbors_year1 = get_nearest_neighbors(embedd_year1, target_word1, n)
        neighbors_year2 = get_nearest_neighbors(embedd_year2, target_word1, n)

        union = get_union(neighbors_year1, neighbors_year2)

        filtered_union = filter_union(union, embedd_year1, embedd_year2, target_word1)
        vec1 = get_second_order_vector(embedd_year1, filtered_union, target_word1)
        vec2 = get_second_order_vector(embedd_year2, filtered_union, target_word1)

        neighbor_words1 = get_nearest_neighbor_words(neighbors_year1)
        neighbor_words2 = get_nearest_neighbor_words(neighbors_year2)

    elif (year1 == year2) and (target_word1 != target_word2):
        results_cosine = 'cosines-'+target_word1+'-'+target_word2+'-n-'+str(n)+'.tsv'

        embedds = SequentialEmbedding.load(corpus, range(year1, year2+10, 10))
        embedd_year = embedds.get_embed(year1)


        neighbors_word1 = get_nearest_neighbors(embedd_year, target_word1, n)
        neighbors_word2 = get_nearest_neighbors(embedd_year, target_word2, n)

        union = get_union(neighbors_word1, neighbors_word2)

        vec1 = get_second_order_vector(embedd_year, union, target_word1)
        vec2 = get_second_order_vector(embedd_year, union, target_word2)

        neighbor_words1 = get_nearest_neighbor_words(neighbors_word1)
        neighbor_words2 = get_nearest_neighbor_words(neighbors_word2)

    cos = get_cosine(vec1, vec2)

    if os.path.isfile(results_dir+results_cosine):
        print('file exists')
        with open(results_dir+results_cosine) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_words, 'w') as outfile1:
        for word1, word2 in zip(neighbor_words1, neighbor_words2):
            outfile1.write(word1+'\t'+word2+'\n')

    with open(results_dir+'/'+results_cosine, 'a') as outfile2:
        result = target_word1+'-'+target_word2+'\t'+str(year1)+'-'+str(year2)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile2.write(result)
    print(cos)

def get_sim_neighbors_ppmi(corpus, target_word1, target_word2, year1, year2, n, results_dir):


    """Two options: either 2 differnt years and 1 target word
    or the same year and 2 target words"""

    if not os.path.isdir(results_dir+'neighbors'):

        os.mkdir(results_dir+'neighbors')

    results_words = 'neighbors/'+target_word1+'-'+target_word2+'-'+str(year1)+'-'+str(year2)+'.tsv'


    if (year1 != year2) and (target_word1 == target_word2):
        results_cosine = 'cosines-'+target_word1+'-n-'+str(n)+'.tsv'

        embedd_year1 = PositiveExplicit.load(corpus+ "/" + str(year1))
        embedd_year2 = PositiveExplicit.load(corpus+ "/" + str(year2))

        with open(corpus+'/'+str(year1)+'-index.pkl', 'rb') as infile:
            year1_vocab = pickle.load(infile, encoding = 'utf-8')
        with open(corpus+'/'+str(year2)+'-index.pkl', 'rb') as infile:
            year2_vocab = pickle.load(infile, encoding = 'utf-8')

        #year1_vocab = pickle.load(open(corpus+'/'+str(year1)+'-index.pkl'))
        #year2_vocab = pickle.load(open(corpus+'/'+str(year2)+'-index.pkl'))

        if (embedd_year1.represent(target_word1).nnz != 0) and (embedd_year2.represent(target_word1).nnz != 0):

            neighbors_year1 = get_nearest_neighbors(embedd_year1, target_word1, n)
            neighbors_year2 = get_nearest_neighbors(embedd_year2, target_word1, n)


            union = get_union(neighbors_year1, neighbors_year2)

            filtered_union = filter_union(union, embedd_year1, embedd_year2, target_word1)

            #clean_union = []

            #for word in union:
            #    if (word in year1_vocab) and (word in year2_vocab):
            #        clean_union.append(word)

            vec1 = get_second_order_vector(embedd_year1, filtered_union, target_word1)
            vec2 = get_second_order_vector(embedd_year2, filtered_union, target_word1)
            #vec1, vec2 = filter_so_vector_for_nans(embedd_year1, embedd_year2, union, target_word1)

            neighbor_words1 = get_nearest_neighbor_words(neighbors_year1)
            neighbor_words2 = get_nearest_neighbor_words(neighbors_year2)

            cos = get_cosine(vec1, vec2)
        else:
            print('word out of vocab')
            cos = 'OOV'
            neighbor_words1 = ['OOV']
            neighbor_words2 = ['OOV']



    elif (year1 == year2) and (target_word1 != target_word2):
        results_cosine = 'cosines-'+target_word1+'-'+target_word2+'-n-'+str(n)+'.tsv'


        embedd_year = PositiveExplicit.load(corpus+ "/" + str(year1))

        if (embedd_year.represent(target_word1).nnz) != 0 and (embedd_year.represent(target_word2).nnz != 0):

            neighbors_word1 = get_nearest_neighbors(embedd_year, target_word1, n)
            neighbors_word2 = get_nearest_neighbors(embedd_year, target_word2, n)

            union = get_union(neighbors_word1, neighbors_word2)

            vec1 = get_second_order_vector(embedd_year, union, target_word1)
            vec2 = get_second_order_vector(embedd_year, union, target_word2)

            neighbor_words1 = get_nearest_neighbor_words(neighbors_word1)
            neighbor_words2 = get_nearest_neighbor_words(neighbors_word2)

            cos = get_cosine(vec1, vec2)
        else:
            print('word out of vocab')
            cos = 'OOV'
            neighbor_words1 = ['OOV']
            neighbor_words2 = ['OOV']

    if os.path.isfile(results_dir+results_cosine):
        print('file exists')
        with open(results_dir+results_cosine) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_words, 'w') as outfile1:
        for word1, word2 in zip(neighbor_words1, neighbor_words2):
            #outfile1.write(word1.encode('utf-8')+'\t'+word2.encode('utf-8')+'\n')
            outfile1.write(word1+'\t'+word2+'\n')

    with open(results_dir+'/'+results_cosine, 'a') as outfile2:
        result = target_word1+'-'+target_word2+'\t'+str(year1)+'-'+str(year2)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile2.write(result)
    print(cos)

#corpus = '/home/pia/Data/embeddings/sgns/coha_word/'
def get_embedding(corpus, year):

    embedd = Embedding.load(corpus+ "/" + str(year))

    return embedd


def get_nearest_neighbors(embedd, target_word, n):

    neighbors = embedd.closest(target_word, n)

    return neighbors

def get_nearest_neighbor_words(neighbors):

    neighbors_words = [sim_word[1] for sim_word in neighbors]

    return neighbors_words


def get_union(neighbors1, neighbors2):

    neighbors_dict1 = dict([(sim_word[1], sim_word[0]) for sim_word in neighbors1])
    neighbors_dict2 = dict([(sim_word[1], sim_word[0]) for sim_word in neighbors2])
    union = list(set(neighbors_dict1.keys()).union(set(neighbors_dict2.keys())))

    return union

def filter_union(union, embedd1, embedd2, target_word):

    filtered_union = []
    for word in union:
        sim1  = embedd1.similarity(word, target_word)
        sim2 = embedd2.similarity(word, target_word)

        if not (math.isnan(sim1) or math.isnan(sim2)):
            filtered_union.append(word)


    return filtered_union
    # This does not work because represent returns this: return csr_matrix((1, len(self.ic))) if the word does not exist

    return filtered_union

def filter_so_vector_for_nans(embedd1, embedd2, union, target_word):

    oov_words1, so_vector1 = find_missing_word_indeces(union, embedd1, target_word)
    oov_words2, so_vector2 = find_missing_word_indeces(union, embedd2, target_word)

    all_indices_of_missing_words = list(sorted( oov_words1+oov_words2))


    for ind in list(reversed(all_indices_of_missing_words)):
        so_vector1.remove(so_vector1[ind])
        so_vector2.remove(so_vector2[ind])

    vec1 = np.array(so_vector1).reshape(1, -1)
    vec_n1 = preprocessing.normalize(vec1, copy=False)

    vec2 = np.array(so_vector2).reshape(1, -1)
    vec_n2 = preprocessing.normalize(vec2, copy=False)

    return vec_n1, vec_n2




def get_second_order_vector(embedd, union, target_word):

    #clean_union = []

    so_vector = [embedd.similarity(target_word, neighbor) for neighbor in union]


    vec = np.array(so_vector).reshape(1, -1)
    vec_n = preprocessing.normalize(vec, copy=False)

    return vec_n




def get_cosine(vec1, vec2):

    cos = vec1.dot(vec2.T)

    return cos.flatten()[0]

if __name__=="__main__":
    corpus = sys.argv[1]
    target_word1 = sys.argv[2]
    target_word2 = sys.argv[3]
    year1 = sys.argv[4]
    year2 = sys.argv[5]
    n = int(sys.argv[6])
    results_dir = sys.argv[7]
    model = sys.argv[8]
    if model == 'sgns':
        get_sim_neighbors(corpus, target_word1, target_word2, year1, year2, n, results_dir)
    elif model == 'svd':
        get_sim_neighbors_svd(corpus, target_word1, target_word2, year1, year2, n, results_dir)
    elif model == 'ppmi':
        get_sim_neighbors_ppmi(corpus, target_word1, target_word2, year1, year2, n, results_dir)

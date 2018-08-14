from sklearn.model_selection import LeaveOneOut
from load_models import load_model, get_nearest_neighbors
from utils import load_data, load_vecs, results_to_file, merge_wi_dicts, check_results
from utils import to_np_array
import numpy as np
from scipy.sparse import csc_matrix
import sys
import glob

def get_centroid(vecs):

    if type(vecs[0]) == np.ndarray:

        matrix = np.array(vecs)
        centroid = np.mean(matrix, axis = 0)

    # sparse matrix
    elif type(vecs[0]) != np.ndarray:

        m = []

        for vec in vecs:
            v_np = vec.toarray()[0]
            m.append(v_np)

        matrix = np.array(m)
        m_sparse = csc_matrix(matrix)
        centroid = m_sparse.mean(axis = 0)
        centroid = csc_matrix(centroid)

    return centroid


def get_centroids_loo(vecs_pos, vecs_neg):

    x = np.array(vecs_pos)

    loo = LeaveOneOut()
    loo.get_n_splits(x)

    centroids = []

    for train_index, test_index in loo.split(x):

        x_train, x_test = x[train_index], x[test_index]



        centroid = get_centroid(x_train)


        centroids.append(centroid)

    # for all negative examples, we can use the full centroid
    full_centroid = get_centroid(x)

    for vec in vecs_neg:
        centroids.append(full_centroid)


    return centroids


def nearest_neighbor_classification_loo(model, feature, n, subset = None):

    predictions = []

    words_pos, words_neg = load_data(feature, subset = subset)

    vecs_pos, wi_dict_pos = load_vecs(model, words_pos)
    vecs_neg, wi_dict_neg = load_vecs(model, words_neg)


    centroids = get_centroids_loo(vecs_pos, vecs_neg)

    words = words_pos + words_neg

    wi_dict = merge_wi_dicts(wi_dict_pos, wi_dict_neg)

    #print(len(words), len(centroids), len(list(wi_dict.keys())))


    for word in words:

        word = word.strip()
        #print(word)

        vec_index = wi_dict[word]

        if vec_index != 'OOV':

            centroid = centroids[vec_index]

            nearest_neighbors = get_nearest_neighbors(model, centroid, n)

            nn_words = [w for c, w in nearest_neighbors]

            #if word == 'pot':
            #    for n in nearest_neighbors:
            #        print(n)

            if word in nn_words:
                predictions.append(1)
            else:
                predictions.append(0)

        else:
            predictions.append('OOV')

    return words, predictions

def nearest_neighbor_classification(model, feature_train, feature_test, n, subset = None):

    predictions = []

    words_pos_train, words_neg_train = load_data(feature_train)

    vecs_pos_train, wi_dict_pos_train = load_vecs(model, words_pos_train)
    vecs_neg_train, wi_dict_neg_train = load_vecs(model, words_neg_train)
    #wi_dict_train = merge_wi_dicts(wi_dict_pos_train, wi_dict_neg_train)

    words_pos_test, words_neg_test = load_data(feature_test)

    vecs_pos_test, wi_dict_pos_test = load_vecs(model, words_pos_test)
    vecs_neg_test, wi_dict_neg_test = load_vecs(model, words_neg_test)
    wi_dict_test = merge_wi_dicts(wi_dict_pos_test, wi_dict_neg_test)


    words_test = words_pos_test + words_neg_test
    x_test = vecs_pos_test + vecs_neg_test


    # transform to np array:

    x_train_pos = to_np_array(vecs_pos_train)
    x_test = to_np_array(x_test)



    centroid = get_centroid(x_train_pos)

    nearest_neighbors = get_nearest_neighbors(model, centroid, n)
    nn_words = [w for c, w in nearest_neighbors]

    #print(centroid)
    #print(nn_words)


    for word in words_test:
        #print(word)
        word = word.strip()
        vec_index = wi_dict_test[word]

        if vec_index != 'OOV':


            if word in nn_words:
                predictions.append(1)
                print(word, 'in nn!')
            else:
                predictions.append(0)
                #print(word)

        else:
            predictions.append('OOV')

    return words_test, predictions




def main():

    experiment_name = 'nearest_neighbors'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    n_begin = int(sys.argv[4])
    n_end = int(sys.argv[5])
    n_step = int(sys.argv[6])
    features = sys.argv[7]
    test = sys.argv[8]



    data = 'data/'

    # indicate data subset (e.g. randomly selected, top similar) by setting
    # subset = rand / top / bottom in load_data
    subset = None

    model = load_model(path_to_model, model_type)

    if n_end == n_begin:
        ns = [n_end]
    else:
        ns = range(n_begin, n_end+n_step, n_step)

    if features == 'all':
        features = [f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*-pos.txt')]
    else:
        features = [features]


    for n in ns:
        for no, feat in enumerate(features):

            if subset:
                par = str(n)+'-'+subset
            else:
                par = n

            print(feat, n, no+1, '/', len(features))

            check = check_results(model_name, experiment_name, feat, par = par)

            if check == False:

                if test == 'loo':
                    par = str(par)+'-loo'
                    words, predictions = nearest_neighbor_classification_loo(model, feat, n, subset = subset)
                    results_to_file(words, predictions, model_name, experiment_name, feat, par=par)
                else:
                    feature_train = feat
                    feature_test = test
                    print(test)
                    words, predictions = nearest_neighbor_classification(model, feature_train, feature_test, n, subset = None)
                    par = str(par) + '-test'
                    results_to_file(words, predictions, model_name, experiment_name, feature_test, par=par)
            else:
                print('results already exist')



if __name__ == '__main__':

    main()

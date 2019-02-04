from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from load_models import load_model
from utils import load_data, load_vecs, results_to_file, merge_wi_dicts
from utils import to_np_array
import numpy as np
from scipy.sparse import csc_matrix
import sys
from random import shuffle
import glob
# unique identifiers for different runs (because of random initialisations)
import datetime

def shuffle_examples(x, y):

    inds = list(range(len(x)))
    shuffle(inds)
    x_shuffled = []
    y_shuffled = []

    for i in inds:
        x_shuffled.append(x[i])
        y_shuffled.append(y[i])
    return np.array(x_shuffled), np.array(y_shuffled), inds


def sparse_to_np_vecs(x):

    if type(x[0]) != np.ndarray:
        x_list = []
        for vec in x:
            x_list.append(vec.toarray()[0])
        x = np.array(x_list)
    else:
        x = np.array(x)

    return x

def create_loo_sets(wi_dict):


    train_test_indices_loo = []
    for word in wi_dict:
        loo_dict_test = dict()
        loo_dict_test[word] = wi_dict[word]
        loo_dict_train = dict()
        for word_other in wi_dict:
            if word_other != word:
                loo_dict_train[word_other] = wi_dict[word_other]
        train_test_indices_loo.append([loo_dict_train, loo_dict_test])

    return train_test_indices_loo



def mlp_classification_loo_new(model, feature):

    words_pos, words_neg = load_data(feature)

    vecs_pos, wi_dict_pos = load_vecs(model, words_pos)
    vecs_neg, wi_dict_neg = load_vecs(model, words_neg)
    wi_dict = merge_wi_dicts(wi_dict_pos, wi_dict_neg)

    words = words_pos + words_neg
    x = vecs_pos + vecs_neg

    x = sparse_to_np_vecs(x)

    y = [1 for vec in vecs_pos]
    [y.append(0) for vec in vecs_neg]

    train_test_indices_loo = create_loo_sets(wi_dict)

    input = len(x[0])
    print('Input size: ', input)

    # Recommended way of setting the nodes in the hidden layer. It is Recommended
    # to start with one hidden layer.
    hidden_layer1_size = int(round((input +1) * (2/3), 0))
    print('Hidden layer size: ', hidden_layer1_size)

    # default solver is adam, but the doc says for smaller data sets, 'lbfgs' performs better.
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size), solver = 'lbfgs')

    loo_predictions = []
    loo_test_words = []
    # For the neural net, the order of examples matters. We shuffle the input:

    for train_test_indices in train_test_indices_loo:

        words_ind_train, words_ind_test = train_test_indices

        x_train_ind = list(words_ind_train.values())
        x_test_ind = list(words_ind_test.values())

        words_test = list(words_ind_test.keys())
        print(x_test_ind)
        print(words_test)
        x_train = [x[ind] for ind in x_train_ind if ind != 'OOV']
        y_train = [y[ind] for ind in x_train_ind if ind != 'OOV']
        x_test = np.array([x[ind] for ind in x_test_ind if ind != 'OOV'])
        print(x_test.shape)
        print(x_test)

        x_train_new, y_train_new, inds_shuffled = shuffle_examples(x_train, y_train)

        mlp.fit(x_train_new, y_train_new)

        if len(x_test) > 0:
            predictions = mlp.predict(x_test)
        else:
            predictions = ['OOV']

        for n, word in enumerate(words_test):
            #vec_index = wi_dict[word]
            loo_test_words.append(word)

            loo_predictions.append(predictions[n])




    return loo_predictions, loo_test_words




def main():

    experiment_name = 'neural_net_classification_loo_new'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    features = sys.argv[4]
    #test = sys.argv[5]


    features = [features]

    model = load_model(path_to_model, model_type)

    sh_par = 'shuff'


    ts = str(datetime.datetime.now()).replace(' ', '-').replace('/', '-').replace('.', '-')
    #par = 'default-'+sh_par+'-'+ts


    for no, feat in enumerate(features):

        print(feat, no+1, '/', len(features))

        par = 'default-'+sh_par+'-'+ts+'-loo'
        predictions, test_words = mlp_classification_loo_new(model, feat)
        results_to_file(test_words, predictions, model_name, experiment_name, feat, par)



if __name__ == '__main__':

    main()

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


def mlp_classification_loo(x, y, shuffle = True):

    loo = LeaveOneOut()
    loo.get_n_splits(x)

    input = len(x[0])
    print('Input size: ', input)

    # Recommended way of setting the nodes in the hidden layer. It is Recommended
    # to start with one hidden layer.
    hidden_layer1_size = int(round((input +1) * (2/3), 0))
    print('Hidden layer size: ', hidden_layer1_size)

    # default solver is adam, but the doc says for smaller data sets, 'lbfgs' performs better.
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size), solver = 'lbfgs')

    predictions = []

    # For the neural net, the order of examples matters. We shuffle the input:

    if shuffle == True:
        x_new, y_new, inds_shuffled = shuffle_examples(x, y)

    elif shuffle == 'reverse':
        x_new = x[::-1]
        y_new = y[::-1]
    else:
        x_new = x
        y_new = y


    for train_index, test_index in loo.split(x_new):
        # All examples
        x_train, x_test = x_new[train_index], x_new[test_index]
        y_train, y_test = y_new[train_index], y_new[test_index]

        mlp.fit(x_train, y_train)
        prediction = mlp.predict(x_test)[0]

        predictions.append(prediction)

    # Map predictions back to original order of examples:

    if shuffle == True:
        predictions_original_order = []

        mapping = []
        for n, i in enumerate(inds_shuffled):
            mapping.append((i, predictions[n]))


        predictions_original_order = [p for i, p in sorted(mapping)]

        for p, po, i, n in zip(predictions, predictions_original_order, inds_shuffled, range(len(inds_shuffled))):

            # check:

            if predictions_original_order[i] != p:
                 print('something went wrong: ', n, po, i, p)

        # additional check:
        # mapp examples back and see if the matrices are the same


    elif shuffle == 'reverse':
        predictions_original_order = predictions[::-1]
    else:
        predictions_original_order = predictions

    return predictions_original_order


def mlp_classification(x_train, y_train, x_test, shuffle = True):


    input = len(x_train[0])
    print('Input size: ', input)

    # Recommended way of setting the nodes in the hidden layer. It is Recommended
    # to start with one hidden layer.
    hidden_layer1_size = int(round((input +1) * (2/3), 0))
    print('Hidden layer size: ', hidden_layer1_size)

    # default solver is adam, but the doc says for smaller data sets, 'lbfgs' performs better.
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size), solver = 'lbfgs')

    predictions = []

    # For the neural net, the order of examples matters. We shuffle the input:

    if shuffle == True:
        x_train_new, y_train_new, inds_shuffled = shuffle_examples(x_train, y_train)

    elif shuffle == 'reverse':
        x_train_new = x_train[::-1]
        y_train_new = y_train[::-1]
    else:
        x_train_new = x_train
        y_train_new = y_train


    mlp.fit(x_train_new, y_train_new)
    predictions = mlp.predict(x_test)

    return predictions


def neural_net_classification_loo(model, feature, shuffle = True):

    final_predictions = []

    words_pos, words_neg = load_data(feature)

    vecs_pos, wi_dict_pos = load_vecs(model, words_pos)
    vecs_neg, wi_dict_neg = load_vecs(model, words_neg)


    words = words_pos + words_neg
    x = vecs_pos + vecs_neg

    # Transform sparse vectors to np vectors

    if type(x[0]) != np.ndarray:
        x_list = []

        for vec in x:
            x_list.append(vec.toarray()[0])

        x = np.array(x_list)

    else:
        x = np.array(x)

    y = [1 for vec in vecs_pos]
    [y.append(0) for vec in vecs_neg]

    y = np.array(y)


    wi_dict = merge_wi_dicts(wi_dict_pos, wi_dict_neg)

    predictions = mlp_classification_loo(x, y, shuffle = shuffle)


    for word in words:

        vec_index = wi_dict[word]

        if vec_index != 'OOV':

            final_predictions.append(predictions[vec_index])

        else:
            final_predictions.append('OOV')

    return words, final_predictions




def neural_net_classification(model, feature_train, feature_test, shuffle = True):

    final_predictions = []

    words_pos_train, words_neg_train = load_data(feature_train)

    vecs_pos_train, wi_dict_pos_train = load_vecs(model, words_pos_train)
    vecs_neg_train, wi_dict_neg_train = load_vecs(model, words_neg_train)
    wi_dict_train = merge_wi_dicts(wi_dict_pos_train, wi_dict_neg_train)


    words_train = words_pos_train + words_neg_train
    x_train = vecs_pos_train + vecs_neg_train
    y_train = [1 for vec in vecs_pos_train]
    [y_train.append(0) for vec in vecs_neg_train]

    y_train = np.array(y_train)


    words_pos_test, words_neg_test = load_data(feature_test)

    vecs_pos_test, wi_dict_pos_test = load_vecs(model, words_pos_test)
    vecs_neg_test, wi_dict_neg_test = load_vecs(model, words_neg_test)
    wi_dict_test = merge_wi_dicts(wi_dict_pos_test, wi_dict_neg_test)


    words_test = words_pos_test + words_neg_test
    x_test = vecs_pos_test + vecs_neg_test
    # Transform sparse vectors to np vectors

    # transform to np array:

    x_train = to_np_array(x_train)
    x_test = to_np_array(x_test)



    predictions = mlp_classification(x_train, y_train, x_test, shuffle = shuffle)


    for word in words_test:

        vec_index = wi_dict_test[word]

        if vec_index != 'OOV':

            final_predictions.append(predictions[vec_index])

        else:
            final_predictions.append('OOV')

    return words_test, final_predictions

def main():

    experiment_name = 'neural_net_classification'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    features = sys.argv[4]
    shuffle = sys.argv[5]
    test = sys.argv[6]

    data = '../data/experiment/'

    print( glob.glob(data+'*-pos.txt'))

    if (features  == 'train') and (test == 'test'):
        features = sorted([f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*_train-pos.txt')])
        test_features = sorted([f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*_test-pos.txt')])

    elif (features == 'all') and (test == 'loo'):
        features = [f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*-pos.txt')\
        if (not 'train' in f) and (not 'test' in f)]

    else:
        features = [features]

    model = load_model(path_to_model, model_type)

    if shuffle == 'shuffle':
        shuffle = True
        sh_par = 'shuff'
    elif shuffle == 'ordered':
        sh_par = 'ordered'
    elif shuffle == 'reverse':
        sh_par = 'reverse'



    ts = str(datetime.datetime.now()).replace(' ', '-').replace('/', '-').replace('.', '-')
    #par = 'default-'+sh_par+'-'+ts


    for no, feat in enumerate(features):

        print(feat, no+1, '/', len(features))

        if test == 'loo':
            par = 'default-'+sh_par+'-'+ts+'-loo'
            words, predictions = neural_net_classification_loo(model, feat, shuffle = shuffle)
            results_to_file(words, predictions, model_name, experiment_name, feat, par)

        else:
            feature_train = feat
            feature_test = test_features[no]
            print(feature_train, feature_test)
            words, predictions = neural_net_classification(model, feature_train, feature_test, shuffle = shuffle)
            par = 'default-'+sh_par+'-'+ts+'-test'
            results_to_file(words, predictions, model_name, experiment_name, feature_test, par)

if __name__ == '__main__':

    main()

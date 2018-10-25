from sklearn.model_selection import LeaveOneOut
import sklearn.linear_model
from load_models import load_model
from utils import load_data, load_vecs, results_to_file, merge_wi_dicts, check_results
from utils import to_np_array
import numpy as np
from scipy.sparse import csc_matrix
import sys
import glob

def lr_classification_loo(x, y):

    loo = LeaveOneOut()
    loo.get_n_splits(x)

    model_lr =  sklearn.linear_model.LogisticRegression()
    predictions = []


    for train_index, test_index in loo.split(x):
        # All examples
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_lr.fit(x_train, y_train)
        prediction = model_lr.predict(x_test)[0]

        predictions.append(prediction)

    return predictions


def lr_classification(x_train, y_train, x_test):


    model_lr =  sklearn.linear_model.LogisticRegression()

    model_lr.fit(x_train, y_train)
    predictions = list(model_lr.predict(x_test))

    return predictions


def logistic_regression_classification_loo(model, feature, subset = None):

    final_predictions = []

    words_pos, words_neg = load_data(feature, subset = subset)

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

    predictions = lr_classification_loo(x, y)


    for word in words:

        vec_index = wi_dict[word]

        if vec_index != 'OOV':

            final_predictions.append(predictions[vec_index])

        else:
            final_predictions.append('OOV')

    return words, final_predictions

def logistic_regression_classification(model, feature_train, feature_test, subset = None):

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

    predictions = lr_classification(x_train, y_train, x_test)


    for word in words_test:

        vec_index = wi_dict_test[word]

        if vec_index != 'OOV':

            final_predictions.append(predictions[vec_index])

        else:
            final_predictions.append('OOV')

    return words_test, final_predictions



def main():

    experiment_name = 'logistic_regression'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    features = sys.argv[4]
    test = sys.argv[5]

    # indicate data subset (e.g. randomly selected, top similar) by setting
    # subset = rand / top / bottom in load_data
    subset = None


    data = '../data/experiment/'


    if (features  == 'train') and (test == 'test'):
        features = sorted([f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*_train-pos.txt')])
        test_features = sorted([f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*_test-pos.txt')])
        print(test_features)

    elif (features == 'all') and (test == 'loo'):
        features = [f.split('/')[-1].split('-')[0] for f in glob.glob(data+'*-pos.txt')\
        if (not 'train' in f) and (not 'test' in f)]

    else:
        features = [features]


    model = load_model(path_to_model, model_type)


    for no, feat in enumerate(features):

        par = 'default'

        print(no, feat)

        if subset:
            par = par+'-'+subset
        else:
            par = par

        print(feat, no+1, '/', len(features))

        check = check_results(model_name, experiment_name, feat, par = par+'-'+test)

        if check == False:

            if test == 'loo':
                words, predictions = logistic_regression_classification_loo(model, feat, subset = subset)
                par = par +'-loo'
                results_to_file(words, predictions, model_name, experiment_name, feat, par=par)
            else:
                feature_train = feat
                feature_test = test_features[no]
                words, predictions = logistic_regression_classification(model, feature_train, feature_test, subset = None)
                par = par + '-test'
                results_to_file(words, predictions, model_name, experiment_name, test, par)
        else:
            print('results already exist')

if __name__ == '__main__':

    main()

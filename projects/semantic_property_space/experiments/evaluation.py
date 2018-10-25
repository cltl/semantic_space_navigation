import os
import glob
from sklearn import metrics
import numpy as np
from utils import load_data, load_gold
import pandas as pd
import sys
from collections import Counter




def get_truth(feature):

    #words_pos, words_neg = load_data(feature, test = test)


    words_pos, words_neg = load_gold(feature)
    words = words_pos + words_neg
    labels = [1 for word in words_pos]
    [labels.append(0) for word in words_neg]

    return labels


def load_results(path):


    with open(path) as infile:
        lines = infile.read().strip().split('\n')

    predictions = [line.split(',')[1] for line in lines]

    return predictions


def evaluate(labels, predictions):

    clean_labels = []
    clean_predictions = []

    if len(labels) == len(predictions):

        oov = 0

        for i, pred in enumerate(predictions):

            if pred != 'OOV':
                clean_predictions.append(int(pred))
                clean_labels.append(labels[i])
            else:
                oov += 1

        f1 = metrics.f1_score(clean_labels, clean_predictions)
        precision = metrics.precision_score(clean_labels, clean_predictions)
        recall = metrics.recall_score(clean_labels, clean_predictions)

    else:
        f1 = -1
        precision = -1
        recall = -1
        oov = -1

    return f1, precision, recall, oov



def evaluate_all(feature):

    #print(feature)

    labels = get_truth(feature)

    #print(labels)

    scores = []

    results_paths  = glob.glob('../results/*/*/*/'+feature+'.txt')

    cols = [ 'f1', 'p', 'r', 'oov']
    indices = []
    values = []

    for path in results_paths:

        predictions = load_results(path)
        name = '-'.join(path.strip('.').split('.')[0].split('/')[1:])
        f1, precision, recall, oov = evaluate(labels, predictions)

        indices.append(name)
        values.append((f1, precision, recall, oov))

    df = pd.DataFrame(values, columns = cols, index = indices).sort_values('f1', ascending = False)

    return df




def main():

    feature = sys.argv[1]
    eval = sys.argv[2]




    dir = '../evaluation/'
    gold = '../gold/'



    if feature == 'all':
        files =  glob.glob(gold+'*-pos.txt')
        features = [f.split('/')[-1].split('.')[0].split('-')[0] for f in files]
    else:
        features = [feature]
    #print(features[0].split('_'))

    #print(eval)


    if eval == 'loo':

        features = [feature for feature in features if feature.split('_')[-1] !=\
         'test' and not feature.split('_')[0].startswith('poly') and not \
         feature.split('_')[0].startswith('ci') ]
        #features = [feature for feature in features if feature.split('_')[-1] !=\
         #'test' and feature.split('_')[0].startswith('poly') ]
        print(features)
    elif eval == 'test':
        print('test')
        features = [feature for feature in features if feature.split('_')[-1] == 'test' and 'poly_' in feature]
        #print(features)



    if not os.path.isdir(dir):
        os.mkdir(dir)

    highest_per_classifier = Counter()

    for feat in features:
        print(feat)

        scores = evaluate_all(feat)
        if len(scores) != 0:
            #print(str(scores['f1'][0])+', '+'-'.join(scores.index[0].split('-')[:-1])+', '+feat)

            highest_per_classifier['-'.join(scores.index[0].split('-')[:-1])] += 1

        #scores.to_csv(dir+feat+'.csv')


        scores.to_csv(dir+feat+'.csv')

    for cl, c in highest_per_classifier.most_common():
        print(cl,'\t',c)


if __name__ == '__main__':
    main()

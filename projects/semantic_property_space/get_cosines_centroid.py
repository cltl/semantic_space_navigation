from load_models import load_model, get_cosine
from utils import load_data, load_vecs, merge_wi_dicts, results_to_file
from nearest_neighbors import get_centroid
import numpy as np
from scipy.sparse import csc_matrix
import sys
import os
import glob

def get_cosines_centroid(model, feature):

    #cosine_dict = dict()

    cosines = []

    words_pos, words_neg = load_data(feature)

    vecs_pos, wi_dict_pos = load_vecs(model, words_pos)
    vecs_neg, wi_dict_neg = load_vecs(model, words_neg)

    centroid = get_centroid(vecs_pos)

    vecs = vecs_pos + vecs_neg
    print(len(vecs_pos), len(vecs_neg), len(vecs))

    words = words_pos + words_neg

    wi_dict = merge_wi_dicts(wi_dict_pos, wi_dict_neg)

    for word in words:

        i = wi_dict[word]

        if i != 'OOV':
            cos = get_cosine(centroid, vecs[i])
            cosines.append(cos)

            #if word == 'king':

            #    nearest_neighbors = get_nearest_neighbors(model, centroid, 250)
            #    for nn in nearest_neighbors:
            #        print(nn)
        else:
            cosines.append(np.NaN)

    return words, cosines, words_pos


def create_negative_subsamples(feature, words, cosines, words_pos):


    cosine_words = []



    for word, cos in zip(words, cosines):

        if (word not in words_pos) and (str(cos) != 'nan'):
            #print(word, cos)
            cosine_words.append((cos, word))


    n_pos = len(words_pos)

    top_neg = [word for cos, word in sorted(cosine_words, reverse=True)[:n_pos]]

    bottom_neg = [word for cos, word in sorted(cosine_words)[:n_pos]]

    return top_neg, bottom_neg




def subsets_to_data(top_neg, bottom_neg, model_name, feature):

    model_dir = 'data/'+model_name+'/'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    with open(model_dir+feature+'-neg-top.txt', 'w') as outfile:
        outfile.write('\n'.join(top_neg))

    with open(model_dir+feature+'-neg-bottom.txt', 'w') as outfile:
        outfile.write('\n'.join(bottom_neg))




def main():

    experiment_name = 'cosine_distances'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    feature = sys.argv[4]

    model = load_model(path_to_model, model_type)

    if len(sys.argv) == 6:
        if sys.argv[5] == test:
            test = True
            data = 'data-test/'
    else:
        test = False
        data = 'data/'

    if feature == 'all':
        files =  glob.glob(data+'*-pos.txt')
        features = [f.split('/')[-1].split('.')[0].split('-')[0] for f in files]
    else:
        features = [feature]




    for feat in features:

        words, cosines, words_pos = get_cosines_centroid(model, feat)
        results_to_file(words, cosines, model_name, experiment_name, feat)

        top_neg, bottom_neg = create_negative_subsamples(feat, words, cosines, words_pos)

        subsets_to_data(top_neg, bottom_neg, model_name, feat)



if __name__ == '__main__':
    main()

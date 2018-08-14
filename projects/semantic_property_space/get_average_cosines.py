from load_models import load_model, get_cosine
from utils import load_data, load_vecs, merge_wi_dicts, results_to_file
import numpy as np
import sys
import os
import glob
from collections import defaultdict
import pandas as pd

def get_av_cosines_pairs(model, feature):

    #cosine_dict = dict()

    cosines = []

    words_pos, words_neg = load_data(feature)

    vecs_pos, wi_dict_pos = load_vecs(model, words_pos)
    vecs_neg, wi_dict_neg = load_vecs(model, words_neg)


    pairs = []

    for word1 in words_pos:
        for word2 in words_pos:
            if  word1 != word2:
                pair = (word1, word2)
                pair_rev = (word2, word1)

                if (pair not in pairs) and (pair_rev not in pairs):
                    pairs.append(pair)



    for word1, word2 in pairs:

        i1 = wi_dict_pos[word1]
        i2 = wi_dict_pos[word2]

        if (i1 != 'OOV') and (i2!= 'OOV'):
            cos = get_cosine(vecs_pos[i1], vecs_pos[i2])
            cosines.append(cos)


        else:
            #cosines.append(np.NaN)
            print(word1, word2, 'not in model')


    if len(cosines) != 0:
        av_cos = sum(cosines)/len(cosines)
    else:
        av_cos = np.NaN

    print(feature, av_cos)

    return av_cos


def collect_av_cosines(features, model):

    feat_cos_dict = defaultdict(list)


    for feature in features:

        av_cos = get_av_cosines_pairs(model, feature)

        feat_cos_dict['feature'].append(feature)
        feat_cos_dict['av_cos'].append(round(av_cos, 4))

    df = pd.DataFrame.from_dict(feat_cos_dict)
    df = df.set_index('feature')

    return df





def main():

    experiment_name = 'cosine_distances_pairs'

    path_to_model = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]
    feature = sys.argv[4]





    test = False
    data = 'data/'
    results = 'results/'

    if feature == 'all':
        files =  glob.glob(data+'*-pos.txt')
        features = [f.split('/')[-1].split('.')[0].split('-')[0] for f in files]
    else:
        features = [feature]

    if os.path.isfile(results+experiment_name+'/'+model_name+'.txt'):

        df_old = pd.read_csv(results+experiment_name+'/'+model_name+'.txt', index_col = 0)

        features_old = df_old.index.values

        features = [feature for feature in features if feature not in features_old]


    if features != []:

        model = load_model(path_to_model, model_type)

        df = collect_av_cosines(features, model)

        if not os.path.isdir(results+experiment_name):
            os.mkdir(results+experiment_name)


        if os.path.isfile(results+experiment_name+'/'+model_name+'.txt'):
            df_final = pd.concat([df_old, df])

            #df_final = df_new.drop_duplicates(keep = 'last')

        else:


            df_final = df


        df = df_final.sort_values(by=['av_cos'])

        df.to_csv(results+experiment_name+'/'+model_name+'.txt')

    else:
        print('all cosines claculated already')



if __name__ == '__main__':
    main()

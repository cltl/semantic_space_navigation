# Analyze the distances of the positive and negative examples and see if the
# classification approaches can correctly classify concepts that further away
# from the centroid than a correctly identified negative example / closer to the
# centroid


from evaluation import load_results, get_truth
from load_models import get_cosine, load_model, get_nearest_neighbors
from utils import load_vecs, merge_wi_dicts, load_data, load_gold
#from nearest_neighbors import get_centroid
import pandas as pd
import glob
import numpy as np
import os
import sys

def load_cosines_centroid(model_name, feature):

    cosine_dict = dict()


    if os.path.isfile('../results/'+model_name+'/cosine_distances/'+feature+'.txt'):


        with open('../results/'+model_name+'/cosine_distances/'+feature+'.txt') as infile:
            lines = infile.read().strip().split('\n')

        for line in lines:
            word, cos, st = line.strip().split(',')

            if cos == 'nan':
                cosine_dict[word] = np.NaN
            else:
                cosine_dict[word] = float(cos)


        return cosine_dict
    else:
        return None



def collect_answers(feature, model_name):

    prediction_dict = dict()

    words_pos, words_neg = load_gold(feature)
    words = words_pos + words_neg
    truth = get_truth(feature)

    print(feature+','+str(len(words_pos))+','+str(len(words_neg)))

    # load all predictions
    results_paths  = glob.glob('../results/'+model_name+'/*/*/'+feature+'.txt')

    prediction_dict['gold'] = truth
    prediction_dict['words'] = words



    for res in results_paths:
        name = 'pred-'+'-'.join(res.split('/')[2:-1])

        if os.path.isfile(res):


            predictions = load_results(res)
            if predictions != None:
                prediction_dict[name] = predictions
            else:
                prediction_dict[name] = ['-' for w in words]

        else:
            prediction_dict[name] = ['-' for w in words]


    return prediction_dict






def create_cosine_prediction_overview(feature, model_name):

    predictions = collect_answers(feature, model_name)


    cosine_dict = load_cosines_centroid(model_name, feature)

    if cosine_dict != None:

        cosines = [str(cosine_dict[word]) for word  in predictions['words']]


        predictions['cosines'] = cosines


        df = pd.DataFrame.from_dict(predictions).set_index('words')

        sorted_df = df.sort_index(axis = 1)
        # sorted_df=unsorted_df.sort_index(axis=1)

        return sorted_df.sort_values('cosines', ascending = False)
    else:
        return '-'






def main():

    model_name = sys.argv[1]
    feature = sys.argv[2]


    print('feature, positive, negative')
    if not os.path.isdir('../analyses/'):
        os.mkdir('../analyses/')


    if feature == 'all':
        print('all')
        files = glob.glob('../gold/*-pos.txt')
        features = [f.split('/')[-1].split('.')[0].split('-')[0] for f in files]

    else:
        features = [feature]


    for feat in features:

        overview = create_cosine_prediction_overview(feat, model_name)

        if type(overview) != str:

            overview.to_csv('../analyses/'+model_name+'-'+feat+'.csv')



if __name__ == '__main__':

    main()

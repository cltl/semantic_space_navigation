
# analyze relation between performance and diversity of features
# diversity of feature = avergage cosine distance from centroid (old version:
# average cosine distance between all possible pairs)

# new version is a bad idea as this depends too much on the number of examples

import pandas as pd
import sys
import glob
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

def load_performance(model_name, experiment_name, par, feature):


    df = pd.read_csv('evaluation/'+feature+'.csv', index_col = 0)

    index = model_name+'-'+experiment_name+'-'+par+'-'+feature

    f1 = df.at[index, 'f1']

    return float(f1)


def get_optimal_nearest_neighbors(model_name, feature):

    df = pd.read_csv('evaluation/'+feature+'.csv', index_col = 0)

    exp = model_name +'-nearest_neighbors-'


    nn_scores = []

    for ind, row in df.iterrows():

        if ind.startswith(exp):

            nn_scores.append((row[0], ind, row))

    highest_nn_score = max(nn_scores)

    #print(feature, highest_nn_score)

    return float(highest_nn_score[0])




def load_diversity(model_name):

    df = pd.read_csv('results/cosine_distances_pairs/'+model_name+'.txt', index_col = 'feature')

    div_dict = df.to_dict()['av_cos']

    #print(div_dict)

    return div_dict


def load_feature_types():

    type_files = glob.glob('cslb_data/feature_types/*.txt')

    type_dict = dict()

    for f in type_files:


        ft = f.split('/')[-1].split('.')[0].strip()

        if ft == 'taxonomic':
            ft = 't'
        elif ft == 'encyclopaedic':
            ft = 'e'
        elif 'visual' in ft:
            ft = 'vp'
        elif 'other' in ft:
            ft = 'op'
        elif ft == 'functional':
            ft = 'f'
        elif ft == 'part':
            ft = 'p'


        with open(f) as infile:
            features = infile.read().strip().split('\n')
        for feature in features:
            feature = feature.replace('-', '_')
            type_dict[feature] = ft

    return type_dict



def collect_performance_diversity(model_name, experiment_names):


    data = 'data/'


    perf_div_dict = defaultdict(list)

    div_dict = load_diversity(model_name)

    type_dict = load_feature_types()



    files =  glob.glob('gold/*.txt')
    features = set([f.split('/')[-1].split('.')[0].split('-')[0] for f in files])

    for feat in features:

        if (not feat.startswith('imp_')) and (not feat.startswith('crowd_')) and\
        (not feat.startswith('poly')) and (not feat.startswith('ci_')):

            div = div_dict[feat]
            ft = type_dict[feat]
            perf_div_dict['cos'].append(round(float(div), 2))
            perf_div_dict['feature'].append(feat)
            perf_div_dict['type'].append(ft)

            for ex in experiment_names:

                if ex == 'nearest_neighbors':
                    par = 'optimal'

                elif ex == 'neural_net_classification':
                    par = 'default-shuff-2018-07-13-09:31:42-012923-loo'

                else:
                    par = 'default-loo'


                print(feat, ex)

                #print(feat, 'filtered')

                if (ex == 'nearest_neighbors') and (par == 'optimal'):
                    f1 = get_optimal_nearest_neighbors(model_name, feat)
                else:
                    f1 = load_performance(model_name, ex, par, feat)

                print(type(f1))
                perf_div_dict['f1-'+ex].append(round(f1, 2))



    print(len(perf_div_dict['cos']))
    print(len(perf_div_dict['f1-nearest_neighbors']))
    print(len(perf_div_dict['f1-neural_net_classification']))
    print(len(perf_div_dict['f1-logistic_regression']))
    df = pd.DataFrame.from_dict(perf_div_dict)



    return df.set_index('feature').sort_values(by = ['cos'])







def main():


    if not os.path.isdir('plots/'):
        os.mkdir('plots/')

    model_name = 'word2vec_google_news'
    experiment_names = ['nearest_neighbors', 'logistic_regression', 'neural_net_classification']
    #par = sys.argv[3]

    df = collect_performance_diversity(model_name, experiment_names)

    #print(df)


    # calculate correlations:
    corr_dict = dict()

    #corr_dict['feature'] = 'spearman-r'

    for ex in experiment_names:

        corr, pv = spearmanr(df['cos'], df['f1-'+ex])
        print(corr, pv)
        corr_dict['f1-'+ex] = round(corr, 2)

    row = pd.Series(corr_dict, name = 'spearman-r')

    df = df.append(row)

    #print(df)

    df.to_csv('plots/table.csv')




    #div_plot = sns.lmplot('div', 'f1', data=df, fit_reg=False, hue = 'type', legend_out = False)

    #div_plot.ax.legend(loc=2)

    #div_plot.ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))

    #div_plot.savefig('plots/'+'div-'+model_name+'-'+experiment_name+'-'+par+'.png')

    #plt.show()

    # calculate correlations

    #spearman_res = spearmanr(df['div', df['f1-nearest_neighbors']])

    #print(spearman_res)




if __name__ == '__main__':
    main()

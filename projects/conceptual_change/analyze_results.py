import glob
import sys
from collections import defaultdict
#import matplotlib.pyplot as plt
import csv
from scipy.stats import spearmanr


def load_results(filepath):

    fieldnames = ['pair', 'year', 'cos']
    with open(filepath) as infile:
        dict_list = list(csv.DictReader(infile, fieldnames = fieldnames, delimiter = '\t'))

    return dict_list


def get_spearman(dict_list, begin = 1900):

    years = [int(d['year']) for d in dict_list if int(d['year']) >= begin]
    cosines = [float(d['cos']) for d in dict_list if int(d['year']) >= begin]

    coeff, pv = spearmanr(years, cosines)
    return coeff, pv


def plot_cosines():
    pass

def correlations_to_file(correlation_dict_list, corpus, model, wordlist1, wordlist2, measure):

    filepath = f'analyses/{corpus}-{model}-{wordlist1}-{wordlist2}-{measure}.csv'
    fieldnames = correlation_dict_list[0].keys()
    with open(filepath, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames = fieldnames)
        writer.writeheader()
        for d in correlation_dict_list:
            writer.writerow(d)



def get_correlations():

    corpus = sys.argv[1]
    model = sys.argv[2]
    wordlist1 = sys.argv[3]
    wordlist2 = sys.argv[4]
    measure = sys.argv[5]

    path_template = f'results/{corpus}/{model}/{wordlist1}-{wordlist2}/{measure}/*.tsv'
    results_paths = glob.glob(path_template)
    
    correlation_dict_list = []

    for p in results_paths:
        corr_dict = dict()
        results_dict_list = load_results(p)
        coeff_1900, pv_1900 = get_spearman(results_dict_list, begin = 1900)
        coeff_1950, pv_1950 = get_spearman(results_dict_list, begin = 1950)
        corr_dict['pair'] = results_dict_list[0]['pair']
        corr_dict['spearman_coeff_1900'] = coeff_1900
        corr_dict['spearman_coeff_1950'] = coeff_1950
        corr_dict['spearman_p_1900'] = pv_1900
        corr_dict['spearman_p_1950'] = pv_1950
        correlation_dict_list.append(corr_dict)


    correlations_to_file(correlation_dict_list, corpus, model, wordlist1, wordlist2, measure)


def main():

    get_correlations()





if __name__ == '__main__':
    main()

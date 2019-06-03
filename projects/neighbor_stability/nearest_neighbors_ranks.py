from dsmtoolkit.load_models import load_model
from dsmtoolkit.load_models import represent
from dsmtoolkit.load_models import get_nearest_neighbors

import sys
import os
import glob


def nn_ranks_to_file(model_paths, year_model, target_word, n):


    results_dir = f'results/{year_model}_ranks'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    init_25_dict = dict()

    for mp in model_paths:
        model = load_model(mp, 'sgns')
        vec = represent(model, target_word)
        init = mp.split('_')[-1]
        if type(vec) != str:
            nn = get_nearest_neighbors(model, vec, n)
            nn_words = [w for cos, w in nn]
            init_25_dict[init] = nn_words[1:26]
            print(len(init_25_dict[init]))
            with open(f'{results_dir}/{target_word}-{str(n)}-{init}.txt', 'w') as outfile:
                outfile.write('\n'.join(nn_words))

    return init_25_dict




def load_nn_ranks(nn_path):

    rank_dict = dict()

    with open(nn_path) as infile:
        nn = infile.read().strip().split('\n')

    for n, w in enumerate(nn):
        rank_dict[w.strip()] = n

    return rank_dict

def get_average_rank_difference(ranks1, ranks2):

    differences = []

    for r1, r2 in zip(ranks1, ranks2):
        differences.append(abs(r1-r2))

    av = sum(differences)/len(differences)
    return av






def main():

    path_to_models = 'models/coha/'

    year = sys.argv[1]
    target_word = sys.argv[2]
    n = 1001


    year_model = f'coha_{str(year)}'

    path_coha_1 = f'{path_to_models}{year_model}/sgns_rand_pinit1/sgns_rand_pinit1'
    path_coha_2 = f'{path_to_models}{year_model}/sgns_rand_pinit2/sgns_rand_pinit2'
    path_coha_3 = f'{path_to_models}{year_model}/sgns_rand_pinit3/sgns_rand_pinit3'

    model_paths = [path_coha_1, path_coha_2, path_coha_3]

    init_25_dict = nn_ranks_to_file(model_paths, year_model, target_word, n)

    average_differences_dict = dict()

    for n1 in range(1,4):
        rank_dict_result = dict()
        init1 = f'pinit{str(n1)}'
        nn25 = init_25_dict[init1]
        ranks1 = list(range(1, len(nn25)+1))
        for n2 in range(1,4):
            if n1 != n2:
                ranks2 = []
                nn_path = f'results/coha_{year}_ranks/{target_word}-{str(n)}-pinit{n2}.txt'
                rank_dict = load_nn_ranks(nn_path)
                for word in nn25:
                    if word in rank_dict:
                        rank_dict_result[word] = rank_dict[word]
                    else:
                        rank_dict_result[word] = '>1000'
                        print('further than 1000')
                    ranks2.append(rank_dict_result[word])

                results_path = f'results/coha_{year}_ranks/{target_word}-pinit{n1}-pinit{n2}-nn25.csv'
                pair = f'pinit{n1}-pinit{n2}'
            
                average_differences_dict[pair] = get_average_rank_difference(ranks1, ranks2)
                with open(results_path, 'w') as outfile:
                    outfile.write(f'pinit{n1},rank in pinit{n2}\n')
                    for word in nn25:
                        rank = rank_dict_result[word]
                        outfile.write(f'{word},{str(rank)}\n')

    average_difference_path = f'results/coha_{year}_ranks/{target_word}-average_rank_differences-nn25.csv'
    with open(average_difference_path, 'w') as outfile:
        outfile.write('init_pair, average_rank_difference\n')
        for pair, difference in average_differences_dict.items():
            outfile.write(f'{pair},{str(difference)}\n')


if __name__ == '__main__':
    main()

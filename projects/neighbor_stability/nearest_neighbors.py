from dsmtoolkit.load_models import load_model
from dsmtoolkit.load_models import represent
from dsmtoolkit.load_models import get_nearest_neighbors

from collections import defaultdict
import os
import sys

def extract_nn_models(model_paths, n, target_word, results_dir):

    results_dict = dict()

    for mp in model_paths:
        model = load_model(mp, 'sgns')
        vec = represent(model, target_word)
        if type(vec) != str:
            nn = get_nearest_neighbors(model, vec, n)
            results_dict[mp] = [w for c, w in nn]
        else:
            results_dict[mp] = []


    return results_dict


def nn_to_file(results_dict, results_dir, target_word, n):

    word_lines = []

    model_paths = results_dict.keys()

    for i in range(len(list(results_dict.values())[0])):
        word_line = []
        for mp in model_paths:
            words = results_dict[mp]
            word_line.append(words[i])
        word_lines.append(word_line)

    with open(f'{results_dir}/{target_word}-{str(n)}.csv', 'w') as outfile:
        outfile.write(','.join(model_paths)+'\n')
        for word_line in word_lines:
            outfile.write(','.join(word_line)+'\n')

def nn_overlaps(results_dict, target_word, n):

    model_paths = results_dict.keys()

    overlap_dict = defaultdict(list)

    all_nns = []

    for wl in results_dict.values():
        all_nns.extend(wl)

    for w in set(all_nns):
        m_count = 0
        for wl in results_dict.values():
            if w in wl:
                m_count += 1
        overlap_dict[m_count].append(w)
    return overlap_dict


def overlaps_to_file(overlap_dict, target_word, n, results_dir):

    with open(f'{results_dir}/{target_word}-nn_overlaps-{str(n)}.csv', 'w') as outfile:
        outfile.write('models shared, words\n')
        for n_models, words in overlap_dict.items():
            outfile.write(f'{str(n_models)},{" ".join(words)}\n')


def main():

    path_to_models = 'models/coha/'

    year = sys.argv[1]
    target_word = sys.argv[2]

    year_model = f'coha_{str(year)}'

    path_coha_1 = f'{path_to_models}{year_model}/sgns_rand_pinit1/sgns_rand_pinit1'
    path_coha_2 = f'{path_to_models}{year_model}/sgns_rand_pinit2/sgns_rand_pinit2'
    path_coha_3 = f'{path_to_models}{year_model}/sgns_rand_pinit3/sgns_rand_pinit3'

    model_paths = [path_coha_1, path_coha_2, path_coha_3]

    n = 25
    results_dir = f'results/{year_model}'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)


    results_dict = extract_nn_models(model_paths, n, target_word, results_dir)
    nn_to_file(results_dict, results_dir, target_word, n)
    overlap_dict = nn_overlaps(results_dict, target_word, n)
    #print(overlap_dict)
    overlaps_to_file(overlap_dict, target_word, n, results_dir)

    pairs = []

    for n1 in range(1,4):
        for n2 in range(1,4):
            pair = f'init{n1}-init{n2}'
            pair_rev = f'init{n2}-init{n1}'
            if n1 != n2 and pair not in pairs and pair_rev not in pairs:

                words1 = set(results_dict[model_paths[n1-1]])
                words2 = set(results_dict[model_paths[n2-1]])
                shared = words1.intersection(words2)
                n_shared = len(shared)
                percentage_shared = n_shared / 25
                #print(words1)
                #print(words2)
                print(pair, n_shared, percentage_shared)
            pairs.append(pair)

    words1 = set(results_dict[model_paths[0]])
    words2 = set(results_dict[model_paths[1]])
    words3 = set(results_dict[model_paths[2]])

    overlap_all = words1.intersection(words2).intersection(words3)
    print('shared across init1-ini2-init3', len(overlap_all), len(overlap_all)/25)






if __name__ == '__main__':
    main()

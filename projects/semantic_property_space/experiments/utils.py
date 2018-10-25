from load_models import load_model
from load_models import represent
import os
import numpy as np

# Load data

def check_results(model_name, experiment_name, feature, par=None):


    results_dir = '../results/'


    model_dir = model_name+'/'

    experiment_name_dir = experiment_name +'/'

    if par:
        par_dir = str(par)+'/'
        dir_list = [results_dir, model_dir, experiment_name_dir, par_dir]
    else:
        dir_list = [results_dir, model_dir, experiment_name_dir]


    dir_str = ''

    for d in dir_list:

        dir_str = dir_str + d

    target = dir_str+feature+'.txt'

    if os.path.isfile(target):
        return True
    else:
        return False


def load_data(feature, subset = None):



    data = '../data/experiment/'

    with open(data+feature+'-pos.txt') as infile:
        words_pos = infile.read().strip().split('\n')

    if subset == None:
        with open(data+feature+'-neg-all.txt') as infile:
            words_neg = infile.read().strip().split('\n')
    elif subset == 'random':
        with open(data+feature+'-neg-rand.txt') as infile:
            words_neg = infile.read().strip().split('\n')

    elif subset == 'closest_model':

        with open(data+model_name+'/'+feature+'-neg-top.txt') as infile:
            words_neg = infile.read().strip().split('\n')

    # Move everything new to gold
    gold_dir = '../gold/'
    if not os.path.isdir(gold_dir):
        os.mkdir(gold_dir)

    with open(gold_dir+feature+'-pos.txt', 'w') as outfile:
        outfile.write('\n'.join(words_pos))
    if subset == None:
        with open(gold_dir+feature+'-neg-all.txt', 'w') as outfile:
            outfile.write('\n'.join(words_neg))
    elif subset == 'random':
        with open(gold_dir+feature+'-neg-rand.txt', 'w') as outfile:
            outfile.write('\n'.join(words_neg))
    elif subset == 'closest_model':
        with open(gold_dir+feature+'-neg-top.txt', 'w') as outfile:
            outfile.write('\n'.join(words_neg))



    return words_pos, words_neg


def load_gold(feature, subset = None):


    data = '../gold/'

    with open(data+feature+'-pos.txt') as infile:
        words_pos = infile.read().strip().split('\n')

    if subset == None:
        with open(data+feature+'-neg-all.txt') as infile:
            words_neg = infile.read().strip().split('\n')
    elif subset == 'random':
        with open(data+feature+'-neg-rand.txt') as infile:
            words_neg = infile.read().strip().split('\n')

    elif subset == 'closest_model':

        with open(data+model_name+'/'+feature+'-neg-top.txt') as infile:
            words_neg = infile.read().strip().split('\n')

    return words_pos, words_neg

def load_vecs(model, words):

    # Map words to index of word vec in matrix or assign 'oov' (for evaluation)
    wi_dict = dict()

    vecs = []

    vec_counter = 0

    for word in words:
        word = word.strip()

        vec = represent(model, word)

        ds = [d for d in vec if type(d) != str]

        if ds:

            wi_dict[word] = vec_counter
            vecs.append(vec)
            vec_counter += 1


        elif vec == 'OOV':


            wi_dict[word] = 'OOV'


    return vecs, wi_dict


def results_to_file(words, predictions, model_name, experiment_name, feature, par=None):


    results_dir = '../results/'


    model_dir = model_name+'/'

    experiment_name_dir = experiment_name +'/'



    if par:
        par_dir = str(par)+'/'
        dir_list = [results_dir, model_dir, experiment_name_dir, par_dir]
    else:
        dir_list = [results_dir, model_dir, experiment_name_dir]


    dir_str = ''

    for d in dir_list:

        dir_str = dir_str + d

        if not os.path.isdir(dir_str):
            os.mkdir(dir_str)





    with open(dir_str+feature+'.txt', 'w') as outfile:

        for word, pred in zip(words, predictions):
            outfile.write(','.join([word, str(pred), '\n']))





# Load vectors positive examples

# Load vectors negative examples


def main():

    #path_w2v = '/Users/piasommerauer/Data/dsm/word2vec/movies.bin'

    #path_hyp_sgns = '/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/sgns_rand_pinit1/sgns_rand_pinit1'

    #path_hyp_svd = '/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/svd/svd'

    path_hyp_ppmi = '/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/pmi/pmi'



    #model = load_model(path_w2v, 'w2v')
    #model = load_model(path_hyp_sgns, 'sgns')
    #model = load_model(path_hyp_svd, 'svd')
    model = load_model(path_hyp_ppmi, 'ppmi')

    words_pos, words_neg = load_data('toy')

    vecs, wi_dict = load_vecs(model, words_pos+words_neg)


    for w, i in wi_dict.items():
        if i != 'OOV':
            print(vecs[i].shape)


def merge_wi_dicts(wi_dict_pos, wi_dict_neg):

    wi_dict = dict()

    list_pos = [(i, w) for w, i in wi_dict_pos.items() if i != 'OOV']

    n_pos_vecs = max(list_pos)[0]

    wi_dict = dict()

    for w, i in wi_dict_pos.items():
        wi_dict[w] = i

    for w, i in wi_dict_neg.items():

        if i != 'OOV':
            wi_dict[w] = n_pos_vecs + i + 1
        else:
            wi_dict[w] = i

    #for w, i in wi_dict.items():
    #    print(w,i)

    return wi_dict


def to_np_array(x):

    if type(x[0]) != np.ndarray:
        x_list = []

        for vec in x:
            x_list.append(vec.toarray()[0])

        x = np.array(x_list)

    else:
        x = np.array(x)

    return x


if __name__ == '__main__':

    main()

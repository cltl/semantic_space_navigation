import glob
import sys
from load_models import load_model, get_nearest_neighbors, represent
from utils import to_np_array
from nearest_neighbors import get_centroid
from utils import load_data, load_vecs

# collect candidates from existing data sets that are good candidates for
# extending the negative examples to make sure they do not just represent a
# particular category. In addition, they should be similar to the positive examples


def food_not_used_in_cooking():

    with open('cslb_data/extracted_cslb/data_cslb_implications_clean/imp_is_used_in_cooking-pos.txt') as infile:
        lines_cooking = infile.read().strip().split('\n')

    with open('cslb_data/extracted_cslb/data_cslb_implications_clean/imp_is_food-pos.txt') as infile:
        lines_food = infile.read().strip().split('\n')


    food_only = []

    for word in lines_food:
        word = word.strip()

        if word not in [word.strip() for word in lines_cooking]:
            food_only.append(word)

    return food_only


def get_false_positives_vanilla_set_up():

    prediction_overviews = glob.glob('analyses/*.csv')


    # exclude all results of already cleaned data sets etc


    for f in prediction_overviews:

        res = f.split('/')[-1].split('-')[-1].split('.')[0]
        par = f.split('/')[-2]
        #print(par, '\n')

        if not res.startswith('imp_') and not res.startswith('poly'):

            #print(res.split('-')[-1])

            with open(f) as infile:
                lines = infile.read().strip().split('\n')


            for line in lines[1:]:
                concept_predictions  = line.split(',')
                concept = concept_predictions[0]
                gold = concept_predictions[2]
                predictions = concept_predictions[3:]
                negatives = [p for p in predictions if p == '1']


                # find all false_negatives:


                if gold == '0' and negatives != []:
                    print(res+','+concept)



def collect_nearest_neighbors_centroid(model, target_feature, n,):


    words_pos_target, words_neg_target = load_data(target_feature)

    vecs_pos_target, wi_dict_pos_target = load_vecs(model, words_pos_target)

    # transform to np array:

    x_target_pos = to_np_array(vecs_pos_target)

    centroid = get_centroid(x_target_pos)

    nearest_neighbors = get_nearest_neighbors(model, centroid, n)
    nn_words = [w for c, w in nearest_neighbors]


    # filter out words already in the input:

    nn_filtered = [word for word in nn_words if word not in words_pos_target]


    return nn_filtered


def collect_nearest_neighbors_word(model, target_word, n):


    target_vec = represent(model, target_word)

    if target_vec[0] != 'O':
        print(target_word)

        nearest_neighbors = get_nearest_neighbors(model, target_vec, n)
        nn_words = [w for c, w in nearest_neighbors]
    else:
        nn_words = []
        print('OOV', target_word)




    return nn_words



def main():

    #food_only = food_not_used_in_cooking()
    #print(food_only)
    # --> this only gives 'sandwich' as a good example
    #get_false_positives_vanilla_set_up()

    path_to_model = '/Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'

    model_type =  'w2v'

    model = load_model(path_to_model, model_type)


    nc = 100
    nw = 100
    #feature_list = [f.split('/')[1].split('.')[0].split('-')[0] for f in glob.glob('data/*.txt')]
    # This is a featured article. Click here for more information. Page semi-protected
    # Hippopotamus

    feature_dict = {
    #'ci_is_dangerous':['gun', 'tiger', 'drug', 'hippopotamus', 'shovel', 'knife' ],
    'ci_is_an_animal': ['moose', 'alligator', 'grasshopper', 'kangaroo']
    #'ci_does_kill': ['gun', 'tiger', 'hippopotamus', 'knife', 'mosquito']
    #'ci_has_wheels':['car', 'sledge', 'ship'],
    #'ci_is_found_in_seas':['squid', 'otter', 'beaver', 'carp'],
    #'ci_is_used_in_cooking':['ladel', 'butter', 'pizza']
    }

    for feat, seed_words in feature_dict.items():

        nn_all = []

        feature = feat[3:]

        nn_centroid = collect_nearest_neighbors_centroid(model, feat, nc)

        with open('data_extension/nearest_neighbor_centroid_word2vec_google_news/'+feat+'-nn'+str(nc)+'-round2.txt', 'w') as outfile:
            for w1 in nn_centroid:
                nn_all.append(w1)
                outfile.write(feature+','+w1+'\n')


        for sw in seed_words:


            nn_word = collect_nearest_neighbors_word(model, sw, nw)

            # filter words from centroid:
            #nn_word_filtered = [sw]

            nn_word_filtered = [word for word in nn_word if word not in nn_centroid]

            print(nn_word_filtered)


            with open('data_extension/nearest_neighbor_seed_words_word2vec_google_news/'+feat+'-'+sw+'-nn'+str(nw)+'-round2.txt', 'w') as outfile1:

                for w2 in nn_word_filtered:
                    print(w2)
                    nn_all.append(w2)
                    outfile1.write(feature+','+w2+'\n')

        with open('data_extension/nearest_neighbor_all_word2vec_google_news/'+feat+'-round2.txt', 'w') as outfile2:
            for w3 in nn_all:
                print(w3)
                outfile2.write(feature+','+w3+'\n')













if __name__ == '__main__':

    main()

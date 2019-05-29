import sys
from sim_pair import get_sim_pair
from sim_pair import get_sim_pair_svd
from sim_pair import get_sim_pair_ppmi
import itertools

def main():

    corpus = sys.argv[1]
    wordsfile1 = sys.argv[2]
    wordsfile2 = sys.argv[3]
    results_dir = sys.argv[4]
    model = sys.argv[5]
    #if corpus == 'coha-word':
    years_range = range(1900, 2000, 10)
    #else:
	#years_range = range(1900, 1990, 10)

    word_list1 = get_word_list(wordsfile1)
    word_list2 = get_word_list(wordsfile2)


    pairs = list(itertools.product(word_list1, word_list2))

    list_set_pairs = [tuple(sorted(list(pair))) for pair in pairs]

    set_pairs = set(list_set_pairs)

    for pair in set_pairs:
        target_word1 = pair[0]
        target_word2 = pair[1]

        if target_word1 != target_word2:
            print(target_word1, target_word2)

            for year in years_range:
                print(year)
                if model == 'sgns':
                    get_sim_pair(corpus, target_word1, target_word2, year, results_dir)
                elif model == 'svd':
                    get_sim_pair_svd(corpus, target_word1, target_word2, year, results_dir)
                elif model == 'ppmi':
                    get_sim_pair_ppmi(corpus, target_word1, target_word2, year, results_dir)





def get_word_list(wordsfile):

    with open(wordsfile) as infile:
        word_list = [word.lower() for word in infile.read().strip().split('\n')]

    return word_list

if __name__=="__main__":
    main()

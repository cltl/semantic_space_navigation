import sys
from nearest_neighbors import get_sim_neighbors
from nearest_neighbors import get_sim_neighbors_svd
from nearest_neighbors import get_sim_neighbors_ppmi
def main():

    wordsfile = sys.argv[1]
    corpus = sys.argv[2]
    results_dir = sys.argv[3]
    model = sys.argv[4]
    n = 25


    word_list = get_word_list(wordsfile)

    for target_word in word_list:
        print(target_word)

        for year in range(1900, 1990, 10):
            year1 = year
            year2 = year+10
            if model == 'sgns':
                get_sim_neighbors(corpus, target_word, target_word, year1, year2, n, results_dir)
            elif model == 'svd':
                get_sim_neighbors_svd(corpus, target_word, target_word, year1, year2, n, results_dir)
            elif model == 'ppmi':
                get_sim_neighbors_ppmi(corpus, target_word, target_word, year1, year2, n, results_dir)






def get_word_list(wordsfile):

    with open(wordsfile) as infile:
        word_list = [word.lower() for word in infile.read().strip().split('\n')]

    return word_list


if __name__=="__main__":
    main()

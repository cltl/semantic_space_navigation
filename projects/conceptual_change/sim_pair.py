from representations.sequentialembedding import SequentialEmbedding
from representations.sequentialembedding import SequentialSVDEmbedding
from representations.explicit import PositiveExplicit
import sys
import os


def get_sim_pair(corpus, target_word1, target_word2, year, results_dir):

    results_pair = target_word1+'-'+target_word2+'-cosines.tsv'

    embedds = SequentialEmbedding.load(corpus, range(year, year+10, 10))
    embedd = embedds.get_embed(year)


    cos = embedd.similarity(target_word1, target_word2)

    if os.path.isfile(results_dir+results_pair):
        print('file exists')
        with open(results_dir+results_pair) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_pair, 'a') as outfile:
        result = target_word1+'-'+target_word2+'\t'+str(year)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile.write(result)

    print(cos)

def get_sim_pair_svd(corpus, target_word1, target_word2, year, results_dir):

    results_pair = target_word1+'-'+target_word2+'-cosines.tsv'

    embedds = SequentialSVDEmbedding.load(corpus, range(year, year+10, 10))
    embedd = embedds.get_embed(year)


    cos = embedd.similarity(target_word1, target_word2)

    if os.path.isfile(results_dir+results_pair):
        print('file exists')
        with open(results_dir+results_pair) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_pair, 'a') as outfile:
        result = target_word1+'-'+target_word2+'\t'+str(year)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile.write(result)

    print(cos)

def get_sim_pair_ppmi(corpus, target_word1, target_word2, year, results_dir):

    results_pair = target_word1+'-'+target_word2+'-cosines.tsv'


    embedd = PositiveExplicit.load(corpus+ "/" + str(year))


    cos = embedd.similarity(target_word1, target_word2)

    if os.path.isfile(results_dir+results_pair):
        print('file exists')
        with open(results_dir+results_pair) as infile:
            existing_results = infile.read().split('\n')

    else:
        existing_results = []

    with open(results_dir+results_pair, 'a') as outfile:
        result = target_word1+'-'+target_word2+'\t'+str(year)+'\t'+str(cos)+'\n'
        if result.strip() in existing_results:
            print('result already there')
        else:
            outfile.write(result)

    print(cos)

if __name__=="__main__":
    corpus = sys.argv[1]
    target_word1 = sys.argv[2]
    target_word2 = sys.argv[3]
    year = sys.argv[4]
    results_dir = sys.argv[5]
    model = sys.argv[6]
    if model == 'sgns':
        get_sim_pair(corpus, target_word1, target_word2, year, results_dir)
    elif model == 'svd':
        get_sim_pair_svd(corpus, target_word1, target_word2, year, results_dir)
    elif model == 'ppmi':
        get_sim_pair_ppmi(corpus, target_word1, target_word2, year, results_dir)

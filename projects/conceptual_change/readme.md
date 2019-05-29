This repository contains code accompanying the following paper:

[This is work in progress.]


@inproceedings{sommerauer2019conceptual,
    title = "Conceptual Change and Distributional Semantic Models:
              an Exploratory Study on Pitfalls and Possibilities",
    author = "Sommerauer, Pia  and Fokkens, Antske",
    booktitle = "to be published in Proceedings of 1st International Workshop
                  on Computational Approaches to Historical Language Change 2019",
    year = "2019",
}

The code used for the experiments is a slightly modified version of the Histwords code used in Hamilton et al. (2016), which can be found here: https://github.com/williamleif/histwords

The code has been updated to python 3.

Please cite the following paper if you use our code:

@article{hamilton2016diachronic,
  title={Diachronic word embeddings reveal statistical laws of semantic change},
  author={Hamilton, William L and Leskovec, Jure and Jurafsky, Dan},
  journal={arXiv preprint arXiv:1605.09096},
  year={2016}
}

To replicate our experiments, please follow these steps:

# 1. Installations & downloads

Clone the repository:

Required packages:

[to be filled in]

Download the distributional models used in Hamilton et al (2016) from here:
https://nlp.stanford.edu/projects/histwords/

# 2. Conceptual system

For this research, we translated the conceptual system of racism to words and similarity/relatedness relations between words. All word lists (corresponding to different components of the conceptual system) are stored in the directory `wordlists/`.

Use the filenames (without the directory path and extension) to carry out the experiments outlined below.

# 3. Experiments

* Pairwise change:

  `sh run_experiments_pair.sh [path_to_histowrds_embeddings] [name_word_list1] [name_word_list2] [model] [corpus]`

e.g.:

  `sh run_experiments_pair.sh /Users/piasommerauer/Data/dsm/histword_models test1 test2 sgns coha-word`


* Nearest neighbor comparison across decades and model types

`sh run_experiments_neighbors.sh [path_to_histowrds_embeddings] [name_word_list] sgns coha-word`

e.g.:

`sh run_experiments_neighbors.sh /Users/piasommerauer/Data/dsm/histword_models test1 ppmi coha-word`

* Nearest neighbor comparison across models:

Compare the nearest neighbors of a word by considering the nearest neighbor lists stored in the results directory.

e.g. The nearest neighbors of the words in test1 for the year 1920 in all three model types trained on COHA can be found in:

`results/coha-word/ppmi/test1/neighbors/neighbors/news-news-1910-1920.tsv `
`results/coha-word/svd/test1/neighbors/neighbors/news-news-1910-1920.tsv `
`results/coha-word/ppmi/test1/neighbors/neighbors/news-news-1910-1920.tsv `

* Nearest neighbor comparison across different initializations of the same model:

Code for these experiments can be found here [link].

# Contact

pia.sommerauer[at]vu.nl and antske.fokkens[at]vu.nl

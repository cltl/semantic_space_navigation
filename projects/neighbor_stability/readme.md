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

Create your own models with different initialization using this repository [fill in link]

without the directory path and extension) to carry out the experiments outlined below.

# 3. Experiments


* Nearest neighbor comparison across different initializations of the same model:

(a) Compare number of shared neighbors among the top 25 neighbors of a target word:

`python nearest_neighbors.py` [year] [target_word]

(b) Analyze the rank differences among the top 25 neighbors of a target word (taking 1001 neighbors into account - number can be modified):

`python nearest_neighbors_ranks.py` [year] [target_word]

Example commands can be found in `get_neighbor_stability.sh`

The remaining experiments discussed in the paper can be found here [link]. 

# Contact

pia.sommerauer[at]vu.nl and antske.fokkens[at]vu.nl

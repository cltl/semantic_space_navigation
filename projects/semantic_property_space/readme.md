# Semantic property space

This repository contains preliminary code to investigate to what extent word embedding vectors contain information about semantic properties. The fundamental assumption is that word embedding vector dimensions can pick up information about some semantic properties from co-occurrance patterns in natural language. We test this by means of a binary supervised classification task in which embedding vectors are used as infput vectors for a supervised classifier which predicts whether a word has a specific semantic property or not. For training and testing, we make use of the  CSLB semantic property norm dataset collected by Devereux et al. (2014) and our own extensions to it by means of crowdsourcing. We compare the results to performance achieved by employing a simple full-vector-cosine-siminarlty based nearest-neighbor approach. The full experimental set-up and datasets extensions are described in the following paper [accepted at the Blackbox.nlp workshop](https://blackboxnlp.github.io/) :

Sommerauer, Pia, and Antske Fokkens. "Firearms and Tigers are Dangerous, Kitchen Knives and Zebras are Not: Testing whether Word Embeddings Can Tell." arXiv preprint arXiv:1809.01375 (2018).


If you make use of the data and/or annotations described in this paper, please also refer to the creators of the CSLB dataset:

Devereux, B.J., Tyler, L.K., Geertzen, J., Randall, B. (2014). The Centre for Speech, Language and the Brain (CSLB) Concept Property Norms. Behavior Research Methods, 46(4), pp 1119-1127. DOI: 10.3758/s13428-013-0420-4.

If you have questions, please contact Pia Sommerauer (pia.sommerauer@live.com) or Antske Fokkens (antske.fokkens@vu.nl). The documentation is still in progress.

## Experiments:

All scripts to run the experiments described in the paper can be found in experiments/.

The directory includes scripts to:

- learn features with logistic regression: `logistic_regression.py`
- learn features with a neural network: `neural_net.py`
- predict features via the nearest neighbor of the centroid of its positive examples: `nearest_neighbors.py`
- calculate the average similarity of all concepts in a dataset: `get_average_cosines.py`
- get the distance of the words in a dataset from the centroid of all positive examples: `get_cosines_centriod.py`




### Running experiments


Example script to run all experiments: `experiments/run_experiments.sh`

Note that the commands for running the standard experiments and the experiments on polysemy datasets differ slightly because of the different ways of evaluating the results.

### Results

The results  are written to:

- Nearest neighbors: `results/[model_name]/nearest_neighbors/[n-evaluation]/[property].txt`
- Logistic regression: `results/[model_name]/logistic_regression/[parameters-evaluation]/[property].txt`
- Neural net: `results/[model_name]/neural_net_classification/[parameters-order-timestamp-evaluation]/[property].txt`

### Evaluation

The classifiers are evaluated using precision, recall, f1 per feature excluding out-of-vocabulary words.

The results of the evaluation are written to:

`evaluation/[property].csv`

The .csv file contains the results of all models and classification approaches.


## Data

We used the semantic property norms collected by Devereux et al. (2014) and extended the dataset via a crowdsourcing task. The original data can be dowlowaded at https://cslb.psychol.cam.ac.uk/propnorms. Our extension of the dataset (including intermediate steps and decisions) will be made available at (insert link).

### Extension

The extension provdies:
* negative examples of properties (e.g. *strawberry* is a negative example of *is_an_animal*)
* an additional verification of the some existing concept-property pairs
* additional concepts (collected using the word2vec google news model)

### Format:

The data are stored in `data/blackbox/standard` and `data/blackbox/polysemy`  in the following format:

Standard (leave-one-out evaluation:
* Positive examples: `[property]-pos.txt`
* Negative examples: `[property]-neg-all.txt`

Polysemy(train - test splits):

* `[property]-train-pos.txt`
* `[property]-test-pos.txt`
* `[property]-train-neg-all.txt`
* `[property]-test-neg-all.txt`


If you would like to gain insights into which examples have been added and verified through which step, please consider the data in the directory `implications/`, `extension-candidates` and `crowd`  and download the original CSLB property dataset.

If you want to run an experiment, move your files into the `data/experiment/` directory. Currenty, the positive and negative examples of the property `is_dangerous` are placed there for testing.




## Hypotheses

Hypotheses about specific semantic properties formulated by the authors (independently from each other in the first stage and combined in a second stage) can be found in hypotheses/.



## Supported models

- Word2vec skip-gram in .bin format (model_type: 'w2v')
- Hyperwords models (Levy & Goldstein 2015):
    - Word2vec skip-gram (model_type: 'sgns')
    - PPMI (model_type: 'ppmi')
    - PPMI reduced with SVD (model_type: 'svd')

## Instructions for feature implication annotation:

Files: cslb_data/extracted_cslb/feature_implications_annotated_antkse.csv, cslb_data/extracted_cslb/feature_implications_annotated_pia.csv

Anntoate:

- features which always correlate: 1 (e.g. is_a_mammal - is_an_animal)
- features which can, but do not necessarily correlate: m (for maybe)
- features which are mutually exclusive: 0
- not applicable: na


Direction for implications: The feature in the column implies the feature in the row. We want to increase the data sets of the row-features by adding concepts associated with the column features. We want to compile better sets of negative examples by using concepts associated with features of which we know that they cannot apply to the positive examples of the target feature.

Ask yourself: Is something that is [column] also [row]? Answer with: always, sometimes, never, does not apply.

Our target features for more experiments are listed in the rows.


## Crowdsourcing set-up

Please consult our paper.



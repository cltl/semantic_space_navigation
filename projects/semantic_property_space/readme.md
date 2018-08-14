# Semantic property space

This repository contains preliminary code to investigate to what extent word embedding vectors contain information about semantic properties. The fundamental assumption is that word embedding vector dimensions can pick up information about some semantic properties from co-occurrance patterns in natural language. We test this by means of a binary supervised classification task in which embedding vectors are used as infput vectors for a supervised classifier which predicts whether a word has a specific semantic property or not. For training and testing, we make use of the  CSLB semantic property norm dataset collected by Devereux et al. (2014) and our own extensions to it by means of crowdsourcing. We compare the results to performance achieved by employing a simple full-vector-cosine-siminarlty based nearest-neighbor approach. The full experimental set-up and datasets extensions are described in the following paper:

[insert reference]


If you cite our paper, please also refer to the creators of the CSLB dataset:

Devereux, B.J., Tyler, L.K., Geertzen, J., Randall, B. (2014). The Centre for Speech, Language and the Brain (CSLB) Concept Property Norms. Behavior Research Methods, 46(4), pp 1119-1127. DOI: 10.3758/s13428-013-0420-4.

If you have questions, please contact Pia Sommerauer (pia.sommerauer@live.com). The documentation is still in progress.

## Experiments:

- learn features with logistic regression: logistic_regression.py
- learn features with a neural network: neural_net.py
- predict features via the nearest neighbor of the centroid of its positive
  examples: nearest_neighbors.py
  
  ### Running experiments
  
  (See example script: example_experiments.sh)


  python logistic_regression.py [path_to_model] [model_name] [model_type] [feature]
  
  python neural_net.py [path_to_model] [model_name] [model_type] [feature]
  
  python nearest_neighbors.py [path_to_model] [model_name] [model_type] [neighbors_n_begin]
  [neighbors_n_end] [neighbors_n_step] [feature]



## Data

We used the semantic property norms collected by Devereux et al. (2014) and extended the dataset via a crowdsourcing task. The original data can be dowlowaded at https://cslb.psychol.cam.ac.uk/propnorms. Our extension of the dataset (including intermediate steps and decisions) will be made available at (insert link).

Instructions:

- store positive and negative examples in data/
- naming convention:
    - [feature]-pos.txt; e.g. fruit_test-pos.txt
    - [feature]-neg-all.txt; e.g. fruit_test.neg-all.txt
- each line should contain a single word


## Hypotheses

Hypotheses about specific semantic properties formulated by the authors (independently from each other in the first stage and combined in a second stage) can be found in hypotheses/.

## Results

Predictions are written to results/[model_name]/[experiment_name]/[parameters]/[feature].txt

e.g. results/word2vec_google_news/nearest_neighbors/100/fruit_test.txt
e.g. results/word2vec_google_news/logistic_regression/default/fruit_test.txt
e.g. results/word2vec_google_news/neural_net/default/fruit_test.txt

## Evaluation

- Evaluation is written to evation/.
- Precision, recall, f1 per feature excluding out-of-vocabulary words
- Train/test folds: Leave-one-out cross-validation
- Results of the evaluation are written to evaluation/[feature].txt



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

(to be filled in)






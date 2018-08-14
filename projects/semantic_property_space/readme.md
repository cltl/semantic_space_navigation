#Instructions for feature implication annotation:

Files: cslb_data/extracted_cslb/feature_implications_annotated_antkse.csv, cslb_data/extracted_cslb/feature_implications_annotated_pia.csv

Anntoate:

- features which always correlate: 1 (e.g. is_a_mammal - is_an_animal)
- features which can, but do not necessarily correlate: m (for maybe)
- features which are mutually exclusive: 0
- not applicable: na


Direction for implications: The feature in the column implies the feature in the row. We want to increase the data sets of the row-features by adding concepts associated with the column features. We want to compile better sets of negative examples by using concepts associated with features of which we know that they cannot apply to the positive examples of the target feature.

Ask yourself: Is something that is [column] also [row]? Answer with: always, sometimes, never, does not apply.

Our target features for more experiments are listed in the rows.

#Code for detecting semantic properties in vector representations

##Experiments:

- learn features with logistic regression: logistic_regression.py
- learn features with a neural network: neural_net.py
- predict features via the nearest neighbor of the centroid of its positive
  examples: nearest_neighbors.py


##Data:

- store positive and negative examples in data/
- naming convention:
    - [feature]-pos.txt; e.g. fruit_test-pos.txt
    - [feature]-neg.txt; e.g. fruit_test.neg.txt


##Running experiments

###Nearest neighbors

python logistic_regression.py [path_to_model] [model_name] [model_type] [feature]

python neural_net.py [path_to_model] [model_name] [model_type] [feature]

python nearest_neighbors.py [path_to_model] [model_name] [model_type] [neighbors_n_begin]
[neighbors_n_end] [neighbors_n_step] [feature]



##Results

Predictions are written to results/[model_name]/[experiment_name]/[parameters]/[feature].txt

e.g. results/word2vec_google_news/nearest_neighbors/100/fruit_test.txt
e.g. results/word2vec_google_news/logistic_regression/default/fruit_test.txt
e.g. results/word2vec_google_news/neural_net/default/fruit_test.txt

##Evaluation

Evaluation is written to...

Precision, recall, f1 per feature excluding out-of-vocabulary words

Train/test folds: Leave-one-out cross-validation

Results of the evaluation are written to evaluation/[feature].txt



##Supported models

- Word2vec skip-gram in .bin format (model_type: 'w2v')
- Hyperwords models (Levy & Goldstein 2015):
    - Word2vec skip-gram (model_type: 'sgns')
    - PPMI (model_type: 'ppmi')
    - PPMI reduced with SVD (model_type: 'svd')


##Additional material

- cslb_data/cslb/ contains the original property norms, extracted data and additional statistics
- cslb_data/crowd_annotation/ contains extensions to the original data with crowd sourcing

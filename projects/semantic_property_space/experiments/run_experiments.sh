
# Path to your locally stored model - using word2vec tpy model on a movie
# review corpus for testing. Model created based on these instructions:
# https://streamhacker.com/2014/12/29/word2vec-nltk/
MODEL_PATH='../models/movies.bin'

# Give your model a name so you can distinguish results
# obtained with different models
MODEL_NAME='word2vec_movies'

# Model type:
# word2vec models (created with gensim): w2v
# Models used in the hyperwords experiments (https://bitbucket.org/omerlevy/hyperwords)
# Word2vec skip gram: sgns
# PPMI-SVD model: svd
# PPMI model (sparse matrix): ppmi
# This parameter is necessary for loading the model correctly
MODEL_TYPE='w2v'


# Other arguments:

# loo: leave-one-out evaluation
# all: all semantic properties in the directory data/experiment/
# Shuffle: shuffle training examples for the neural net classifier


### download test model

# Create a toymodel for testing the code:
python create_toy_model.py

#### standard experiments

# Standard experiments are run on all files in the directory 'data/experiment/'
# that DO NOT have 'train' or 'test' in the filename. They are evaluated using
# leave-one-out (loo):


# NN
python nearest_neighbors.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE 100 1000 100 all loo

# LR
python logistic_regression.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all loo

# Neural Net
python neural_net.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all shuffle loo

#### polysemy experiments

# Polysemy experiments are run on all files in the directory 'data/experiment'
# that have 'train' or 'test' in their filename. They are evaluated using a
# test split:


# NN
python nearest_neighbors.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE 100 1000 100 train test

# LR
python logistic_regression.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE train test

# Neural Net
python neural_net.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE train shuffle test


#### Evaluation

python evaluation.py all loo
python evaluation.py all test



#### Cosine distances

# Get cosine distances to centroid
python get_cosines_centroid.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all

# Get average similarities
python get_average_cosines.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all


#### Distance analysis

python create_prediction_overview.py $MODEL_NAME all

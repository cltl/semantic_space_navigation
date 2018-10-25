
# Path to your locally stored model - using word2vec tpy model on a movie
# review corpus for testing. Model created based on these instructions:
# https://streamhacker.com/2014/12/29/word2vec-nltk/
MODEL_PATH='/Users/piasommerauer/Data/dsm/word2vec/movies.bin'

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

#### Experiments

# NN
python nearest_neighbors.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE 100 1000 100 all loo

# LR
python logistic_regression.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all loo

# Neural Net
python neural_net.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all shuffle loo


#### Evaluation


python evaluation.py all loo



#### Cosine distances

# Get cosine distances to centroid
python get_cosines_centroid.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE all

# Get average similarities
python get_average_cosines.py $MODEL_PATH $MODEL_NAME $MODEL_TYPE w2v all


#### Distance analysis

python create_prediction_overview.py $MODEL_NAME all

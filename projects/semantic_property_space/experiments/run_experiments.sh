#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 all loo


###### Experiments

# NN
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/movies.bin word2vec_movies w2v 100 1000 100 all loo - works
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 all loo

# LR
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/movies.bin word2vec_movies w2v all loo - works
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all loo

# Neural Net
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/movies.bin word2vec_movies w2v all shuffle loo - works
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all shuffle loo



##### Evaluation


#python evaluation.py all loo - works



### Cosine distances

#python get_cosines_centroid.py /Users/piasommerauer/Data/dsm/word2vec/movies.bin word2vec_movies w2v all
#python get_cosines_centroid.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all

### Get average similarities
#python get_average_cosines.py /Users/piasommerauer/Data/dsm/word2vec/movies.bin word2vec_movies w2v all - works
#python get_average_cosines.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all


### Distance analysis

#python create_prediction_overview.py word2vec_movies all
#python create_prediction_overview.py word2vec_google_news all

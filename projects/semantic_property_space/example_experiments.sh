

# Model paths pia local:

#'/Users/piasommerauer/Data/dsm/word2vec/movies.bin'
#'/Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'
#'/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/sgns_rand_pinit1/sgns_rand_pinit1'
#'/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/svd/svd'
#'/Users/piasommerauer/Data/dsm/hyperwords/wiki_300Mtok_rec/pmi/pmi'


# Model names used by pia:

#model_name  = 'word2vec_google_news'
#model_name  = 'word2vec_movies'
#model_name  = 'sgns_300M_pinit1'
#model_name  = 'svd_300M'
#model_name  = 'ppmi_300M'


## Extract data from cslb vanilla way:

#python generate_data_cslb_vanilla.py 20 2

# copy files to data:

#cp cslb_data/extracted_cslb/examples_20_2/*.txt data/.


## Extract cosine distances:


#python get_cosines_centroid.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all
#python get_average_cosines.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all


### Logistic regression classification

## all datasets in data/ with loo evaluation:
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all loo

## specific, set in data/ set with train/test splits (in separate files):
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_food_train poly_is_food_test
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_an_animal_train poly_is_an_animal_test
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_dangerous_train poly_is_dangerous_test
#python logistic_regression.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_does_kill_train poly_does_kill_test


### Neural network classification:

## all datasets in data/ with loo evaluation:
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all shuffle loo
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all shuffle loo
#python3 neural_net.py /data/w2v_google/GoogleNews-vectors-negative300.bin word2vec_google_news w2v all shuffle loo


## specific, set in data/ set with train/test splits (in separate files):
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_food_train shuffle poly_is_food_test
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_an_animal_train shuffle poly_is_an_animal_test
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_dangerous_train shuffle poly_is_dangerous_test
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_does_kill_train shuffle poly_does_kill_test
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_is_dangerous_train shuffle poly_is_dangerous_test
#python neural_net.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v poly_does_kill_train shuffle poly_does_kill_test
### Nearest neighbors

## all datasets in data/ with loo evaluation:
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 all loo
#python3 nearest_neighbors.py /data/w2v_google/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 400 1000 100 all loo


## specific, set in data/ set with train/test splits (in separate files):
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 1000 10000 100 poly_is_food_train poly_is_food_test
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 poly_is_an_animal_train poly_is_an_animal_test
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 poly_is_dangerous_train poly_is_dangerous_test
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 poly_does_kill_train poly_does_kill_test
#python nearest_neighbors.py /Users/piasommerauer/Data/dsm/word2vec/GoogleNews-vectors-negative300.bin word2vec_google_news w2v 100 1000 100 poly2_is_an_animal poly2_is_an_animal_test

### Evaluation

# Evaluate all experiments conducted with a feature
python evaluation.py all test
#python evaluation.py all test


### Distance analysis

python create_prediction_overview.py word2vec_google_news all

### Diversity vs performance:

#python plot_performance_diversity.py word2vec_google_news neural_net_classification default-reverse-2018-07-05-14/31/24-069647

#

# word2vec_c adaptation

* Using this code:

There are two steps involved in creating a model for this code:

#1 initiate embeddings for word2vec
./initiate_word2vec.sh $MODELDIR/counts.words.vocab $MODELDIR/pinit1

#2 create and evaluate word2vec with all 3 random initiations and svd
./create_word2vec_with_init.sh pairs voc init out size neg iter

For instance:

./initiate_word2vec.sh data/counts.words.vocab model/pinit1
# settings: size=500, neg=1, iters=50
./create_word2vec_with_init.sh data/pairs data/counts.words.vocab model/pinit1 model/sgns_rand_pinit1 500 1 50



* Provenance:

The word2vec implementation allows you to select a specific initialization. This option is provided to replicate previous experiments and try out variations in settings while controling for the initialization.
The code is an adaptation of Omer Levy and Yoav Goldberg used in the following paper:

"Dependency-Based Word Embeddings". Omer Levy and Yoav Goldberg. ACL 2014.

The adaptations were implemented by Minh Le.

# create a toy w2v model
# based on https://streamhacker.com/2014/12/29/word2vec-nltk/
import os
from gensim import models
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews

if not os.path.isdir('../models/'):
    os.mkdir('../models')


mr = Word2Vec(movie_reviews.sents())

mr.wv.save_word2vec_format('../models/movies.bin', binary = True)

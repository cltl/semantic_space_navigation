import utils
import sys

dm_mat, word_to_i, i_to_word = utils.load_space(sys.argv[1])

print "Testing space..."
if all(i in word_to_i for i in ["software","internet","museum"]):
    print "Test cosine between 'software' and 'internet':",utils.cosine_similarity(dm_mat[word_to_i["software"]], dm_mat[word_to_i["internet"]])
    print "Test cosine between 'software' and 'museum':",utils.cosine_similarity(dm_mat[word_to_i["software"]], dm_mat[word_to_i["museum"]])
else:
    print "Test words not in space. Quitting."

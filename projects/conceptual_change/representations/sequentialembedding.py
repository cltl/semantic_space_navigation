import collections
import random

from representations.embedding import Embedding, SVDEmbedding

class SequentialEmbedding:
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds

    @classmethod
    def load(cls, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = Embedding.load(path + "/" + str(year), **kwargs)
        return SequentialEmbedding(embeds)

    def get_embed(self, year):
        return self.embeds[year]

    def get_subembeds(self, words, normalize=True):
        embeds = collections.OrderedDict()
        for year, embed in self.embeds.iteritems():
            embeds[year] = embed.get_subembed(words, normalize=normalize)
        return SequentialEmbedding(embeds)

    def get_time_sims(self, word1, word2):
       time_sims = collections.OrderedDict()
       for year, embed in self.embeds.iteritems():
           time_sims[year] = embed.similarity(word1, word2)
       return time_sims

    def get_seq_neighbour_set(self, word, n=3):
        neighbour_set = set([])
        for embed in self.embeds.itervalues():
            closest = embed.closest(word, n=n)
            for _, neighbour in closest:
                neighbour_set.add(neighbour)
        return neighbour_set

# Added by Pia:
    def get_time_neighbors(self, word, n):
        neighbour_dict = collections.OrderedDict()

        for year, embed in self.embeds.iteritems():
            neighbour_dict[year] = embed.closest(word, n=n)

        return neighbour_dict

#Added by Pia
    def get_neighbors_within_radius(self, word, radius = 0.5):

        neighbors_within_radius_dict = collections.OrderedDict()

        for year, embed in self.embeds.iteritems():
            neighbors_within_radius_dict[year] = embed.neighbors_within_radius(word, radius = radius)

        return neighbors_within_radius_dict




    def get_seq_closest(self, word, start_year, num_years=10, n=10):
        closest = collections.defaultdict(float)
        for year in range(start_year, start_year + num_years):
            embed = self.embeds[year]
            year_closest = embed.closest(word, n=n*10)
            for score, neigh in year_closest.iteritems():
                closest[neigh] += score
        return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]

    def get_word_subembeds(self, word, n=3, num_rand=None, word_list=None):
        if word_list == None:
            word_set = self.get_seq_neighbour_set(word, n=n)
            if num_rand != None:
                word_set = word_set.union(set(random.sample(self.embeds.values()[-1].iw, num_rand)))
            word_list = list(word_set)
        year_subembeds = collections.OrderedDict()
        for year,embed in self.embeds.iteritems():
            year_subembeds[year] = embed.get_subembed(word_list)

        # Print statements by Pia to find out what is what
        #print similarity(year_subembeds[1960].represent(word), year_subembeds[1970].represent(word))
        #for year, embed in year_subembeds.iteritems():
            #print(year, embed)
        # modified by Pia
        return year_subembeds
        # Original code
        #return SequentialEmbedding.from_ordered_dict(year_subembeds)

    def year_embeddings(self, year1, year2):

        year1_embeddings = self.embeds[year1]
        year2_embeddings = self.embeds[year2]

        return year1_embeddings, year2_embeddings


class SequentialSVDEmbedding(SequentialEmbedding):

    def __init__(self, path, years, **kwargs):
        self.embeds = collections.OrderedDict()
        for year in years:
            self.embeds[year] = SVDEmbedding(path + "/" + str(year), **kwargs)

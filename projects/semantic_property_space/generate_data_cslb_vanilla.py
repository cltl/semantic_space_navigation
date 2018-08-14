import pandas as pd
import sys
import os
import random
# Select only features of at least 20 concepts

# Write reduced matrix to file

# Write feature concept pairs to positive and negative files


def load_select_data(cutoff_concepts, cutoff_pf):

    #with open('cslb_data/raw_cslb/feature_matrix.dat') as infile:
    #    lines = infile.read().strip().split()

    #print(lines[0])


    df = pd.read_csv('cslb_data/raw_cslb/feature_matrix.dat', delimiter = '\t',\
            index_col = 'Vectors')

    print(len(list(df.columns)), len(df))

    for feat, pf in df.items():

        pos = len([x for x in pf if x > cutoff_pf])
        if pos < cutoff_concepts:

            df.drop(feat, axis = 1, inplace = True)
            #print(dropped)

    print(len(list(df.columns)), len(df))

    for c, pf in df.iterrows():
        pos = len([x for x in pf if x > cutoff_pf])
        if pos == 0.0:
            df.drop(c, axis = 0, inplace = True)

    print(len(list(df.columns)), len(df))


    return df


def extract_positive_negative_examples(reduced_df, cutoff_concepts, cutoff_pf):

    path = 'cslb_data/extracted_cslb/examples_'+str(cutoff_concepts)+'_'+str(cutoff_pf)+'/'

    if not os.path.isdir(path):
        os.mkdir(path)



    for feat in reduced_df.columns:
        pos = []
        neg = []
        for concept, values in reduced_df.iterrows():
            val = reduced_df.at[concept, feat]

            # remove disambiguation from concept (doesn't help in the space)
            # later, we can exclude these cases or manipulate training/test data
            # by adding/removing them.

            if '_(' in concept:
                concept = concept.split('(')[0].rstrip('_')

            if val > 0:
                pos.append(concept)
            else:
                neg.append(concept)


            # random subsample of negative examples of the same length as the
            # list of positive examples:

        rand_neg = random.sample(neg, len(pos))

        with open(path+feat+'-pos.txt', 'w') as outfile:
            outfile.write('\n'.join(pos))

        with open(path+feat+'-neg-all.txt', 'w') as outfile:
            outfile.write('\n'.join(neg))

        with open(path+feat+'-neg-rand.txt', 'w') as outfile:
            outfile.write('\n'.join(neg))



def spreadsheet_selected(reduced_df, selected):


    features = reduced_df.columns

    with open('cslb_data/extracted_cslb/feature_implications_annotated_antske.csv', 'w') as outfile:
        outfile.write(' ,'+','.join(features)+'\n')

        for f in selected:
            outfile.write(f+'\n')

    with open('cslb_data/extracted_cslb/feature_implications_annotated_pia.csv', 'w') as outfile:
        outfile.write(' ,'+','.join(features)+'\n')

        for f in selected:
            outfile.write(f+'\n')









def main():

    cutoff_concepts = int(sys.argv[1])
    cutoff_pf = int(sys.argv[2])

    reduced_df = load_select_data(cutoff_concepts, cutoff_pf)

    name = 'cslb_subset_'+str(cutoff_concepts)+'_'+str(cutoff_pf)

    #reduced_df.to_csv('cslb_data/extracted_cslb/'+name+'.csv', sep = '\t')

    #extract_positive_negative_examples(reduced_df, cutoff_concepts, cutoff_pf)

    selected = ['is_animal', 'is_food', 'is_dangerous', 'does_kill', \
    'has_wheels', 'is_black', 'is_found_in_seas', 'is_long', 'is_red',\
    'is_used_in_cooking', 'is_warm', 'is_yellow', 'made_of_wood']

    spreadsheet_selected(reduced_df, selected)

if __name__ == '__main__':
    main()

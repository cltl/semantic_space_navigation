import glob


def remove_duplicates():

    clean_files = glob.glob('nearest_neighbor_all_word2vec_google_news/*clean.txt')

    #unexteded_files = glob.glob('../data/*.txt')


    for file in clean_files:




        # get original data:


        with open(file) as infile:
            lines = infile.read().strip().split()

        clean_lines = []
        for line in lines:

            line_clean = [item.strip() for item in line.split(',')]

            if line_clean not in clean_lines:
                clean_lines.append(line_clean)

        print(file, len(clean_lines))
        with open(file.split('.')[0]+'-no_duplicates.txt', 'w') as outfile:
            for l in clean_lines:
                outfile.write(','.join(l)+'\n')


def check_for_duplicates_old_new():

    old_files = glob.glob('nearest_neighbor_all_word2vec_google_news/*.txt')
    new_files = glob.glob('nearest_neighbor_all_word2vec_google_news/*round2.txt')

    for nf in new_files:
        #print(nf)




        for fo in old_files:

            if ('clean' not in fo and 'round2' not in fo):
                print(fo)




                name_new = nf.split('-round2')[0]
                old_name = fo.split('_clean-no-duplicates')[0]

                new_only = []

                if old_name.startswith(name_new):

                    with open(nf) as infile:
                        new_words = infile.read().strip().split('\n')

                    with open(fo) as infile:
                        old_words = infile.read().strip().split('\n')

                    for wn in new_words:

                        if wn not in old_words:
                            new_only.append(wn)



                    with open(nf.split('.')[0]+'new_only.txt', 'w') as outfile:
                        outfile.write('\n'.join(new_only))







def compare_original_dataset():

    extended =  glob.glob('nearest_neighbor_all_word2vec_google_news/*clean-no_duplicates.txt')

    original = glob.glob('../data/*.txt')


    # load all original pairs:

    all_existing_pairs = []

    for f in original:

        feat = f.split('/')[-1].split('.')[0][3:].split('-')[0]
        #print(f, feat)

        with open(f) as infile:
            lines = infile.read().split('\n')

        for line in lines:

            all_existing_pairs.append((feat, line.strip()))


    final_clean_extended = []
    for ef in extended:
        #print(ef)

        with open(ef) as infile:

            pairs = infile.read().strip().split('\n')


        for pair in pairs:

            feat, word = [p.strip() for p in pair.split(',')]


            new_pair = (feat, word)



            if new_pair not in all_existing_pairs:
                final_clean_extended.append(new_pair)

            if feat == 'is_an_animal':

                addintional_pair = ('is_food', word)

                if addintional_pair not in all_existing_pairs:
                    final_clean_extended.append(addintional_pair)

    final_clean_extended = set(final_clean_extended)
    print(len(final_clean_extended))
    with open('set_for_crowd_extension.csv', 'w') as outfile:
        for p in final_clean_extended:
            outfile.write(','.join(p)+'\n')










def main():

    #check_for_duplicates_old_new()

    remove_duplicates()

    compare_original_dataset()

if __name__ == '__main__':
    main()

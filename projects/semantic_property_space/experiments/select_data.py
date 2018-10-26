# select data for an experiment and move them to the ../data/experiment/ directory
import sys
import glob
from collections import defaultdict
import pandas as pd

def select_data_standard(property_list):

    property_df = pd.read_csv('../data/blackbox_dataset.csv')

    selected_data_dict = defaultdict(list)

    for index, line in property_df.iterrows():
        property = line['property']
        if property in property_list:
            concept = line['concept']
            label = line['label']

            if label == 1:
                selected_data_dict[property+'-pos.txt'].append(concept)

            elif label == 0:
                selected_data_dict[property+'-neg-all.txt'].append(concept)

    return selected_data_dict



def data_to_file(selected_data_dict):

    for filename, concepts in selected_data_dict.items():

        with open('../data/experiment/'+filename, 'w') as outfile:
            outfile.write('\n'.join(concepts))




def main():


    property_list = sys.argv[1:]
    selected_data_dict = select_data_standard(property_list)

    data_to_file(selected_data_dict)


if __name__ == '__main__':
    main()

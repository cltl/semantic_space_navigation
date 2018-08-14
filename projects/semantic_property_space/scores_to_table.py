import glob
from collections import defaultdict
import pandas as pd

def collect_div():


    with open('results/cosine_distances_pairs/word2vec_google_news.txt') as infile:

        lines = infile.read().strip().split('\n')

    div_dict = dict()

    for line in lines[1:]:

        feat = line.split(',')[0]
        div = round(float(line.split(',')[1]), 2)

        div_dict[feat] = div

    return div_dict

def collect_f1_scores(file_list):

    div_dict = collect_div()

    performance_dict = defaultdict(list)
    for f in file_list:

        prop = f.split('/')[-1].split('.')[0]


        with open(f) as infile:
            lines = infile.read().strip().split('\n')



        nns = []
        nets = []
        if (len(lines) > 1) and 'train' not in f:
            performance_dict['av-cos'].append(div_dict[prop])
            for line in lines[1:]:
                print(f)

                line_list = line.split(',')
                system = line_list[0]
                f1 = round(float(line_list[1]), 2)
                system_short = system[21:-len(prop)-5]


                if 'nearest_neighbors' in system_short:

                    nns.append(f1)

                elif 'neural_net' in system_short:
                    nets.append(f1)

                elif 'logistic_regression' in system_short:
                    system_name = 'lr'
                    performance_dict['lr'].append(f1)




            performance_dict['property'].append(prop[3:])
            performance_dict['nn'].append(max(nns))
            performance_dict['net1'].append(nets[0])
            performance_dict['net2'].append(nets[1])

    for k, v in performance_dict.items():
        print(k, len(v))


    return pd.DataFrame.from_dict(performance_dict)

            #for n,l in enumerate(system):
            #    print(n, l)


            #if 'neural_net' in system:
            #    performance_dict[]





def main():

    file_list_ci = glob.glob('evaluation/crowd_*.csv')

    df = collect_f1_scores(file_list_ci).set_index('property')

    latex = df.to_latex()
    print(latex)

if __name__ == '__main__':

    main()

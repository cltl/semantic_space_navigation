

with open('list-of-ethnic-slurs-filtered-annotation-antske.csv', 'r') as infilea:
    labelsa = [line.split('\t')[0] for line in infilea.read().split('\n')[1:]]

     

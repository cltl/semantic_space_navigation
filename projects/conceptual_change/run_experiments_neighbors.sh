start=$(date "+%F-%T")

#UUID='run'.$start

EMBEDDINGS=$1
WORDS1=$2 #(file name of word list without extension)
#WORDS2=$4 #(file name of word list without extension)
MODEL=$3
CORPUS=$4

MEASURE=neighbors
RESULTS=results

IN=$EMBEDDINGS/$MODEL/$CORPUS
WORDLIST1=word_lists/$WORDS1.txt
#WORDLIST2=word_lists/$WORDS2.txt
mkdir $RESULTS
mkdir $RESULTS/$CORPUS
mkdir $RESULTS/$CORPUS/$MODEL
mkdir $RESULTS/$CORPUS/$MODEL/$WORDS1
mkdir $RESULTS/$CORPUS/$MODEL/$WORDS1/$MEASURE
RESDIR=$RESULTS/$CORPUS/$MODEL/$WORDS1/$MEASURE/

python experiments_same_word_neighbors.py $WORDLIST1 $IN $RESDIR $MODEL

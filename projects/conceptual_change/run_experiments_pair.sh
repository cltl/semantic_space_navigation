
start=$(date "+%F-%T")

#UUID='run'.$start

EMBEDDINGS=$1
WORDS1=$2 #(file name of word list without extension)
WORDS2=$3 #(file name of word list without extension)
MODEL=$4
CORPUS=$5
MEASURE='pair'

IN=$EMBEDDINGS/$MODEL/$CORPUS
WORDLIST1=word_lists/$WORDS1.txt
WORDLIST2=word_lists/$WORDS2.txt

RESULTS=results

mkdir $RESULTS
mkdir $RESULTS/$CORPUS
mkdir $RESULTS/$CORPUS/$MODEL
mkdir $RESULTS/$CORPUS/$MODEL/$WORDS1-$WORDS2
mkdir $RESULTS/$CORPUS/$MODEL/$WORDS1-$WORDS2/$MEASURE
RESDIR=$RESULTS/$CORPUS/$MODEL/$WORDS1-$WORDS2/$MEASURE/


python experiments_sim_pair.py $IN $WORDLIST1 $WORDLIST2 $RESDIR $MODEL

python analyze_results.py $CORPUS $MODEL $WORDS1 $WORDS2  $MEASURE

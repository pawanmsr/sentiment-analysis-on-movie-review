#!/bin/bash

set -e

FILE=glove.6B/glove.6B.50d.txt
exists=false
if test -f "$FILE"
then
    exists=true
    read -p "50d GloVe exists. Redownload? [y/n]: " ans
    if [ $ans == 'y' ] || [ $ans == 'Y' ]
    then
        exists=false
    fi
fi

if [[ "$exists" == false ]]
then
    bash get-glove.sh
fi

eval "$(conda shell.bash hook)"
read -p "enter python3 conda environment name: " env

conda activate $env
conda install numpy keras nltk scikit-learn pandas
conda install matplotlib graphviz python-graphviz
pip install pydot
python -m nltk.downloader popular
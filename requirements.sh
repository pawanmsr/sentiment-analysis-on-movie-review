#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
read -p 'enter python3 conda environment name: ' env

conda activate $env
conda install matplotlib numpy keras nltk scikit-learn pandas
python -m nltk.downloader popular
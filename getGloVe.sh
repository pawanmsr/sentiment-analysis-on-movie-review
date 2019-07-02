#!/bin/bash

GLOVE_DIR="glove.6B"
mkdir -p $GLOVE_DIR
cd $GLOVE_DIR

# Get GloVe vectors
if hash wget 2>/dev/null; then
  wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
else
  curl -O http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
fi
unzip glove.6B.zip
rm glove.6B.zip

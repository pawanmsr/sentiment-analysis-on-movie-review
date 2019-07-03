
import os
import re
import sys
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import visualize

DATA_DIR = "sentiment-analysis-on-movie-reviews/"
RESULT_DIR = "results/"
MODEL_DIR = 'models/'

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

sentiment_labels = {0 : 'negative', 1 : 'somewhat negative', 2 : 'neutral', 3 : 'somewhat positive', 4 : 'positive'}

def preprocess_dataframe(filename):
    df = pd.read_csv(DATA_DIR + filename, sep = '\t', index_col = 'PhraseId')
    df['Phrase'] = df['Phrase'].str.lower()
    df['Phrase'] = df['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9]', ' ', x)))
    
    savename = filename.split('.')[0]
    # df.to_csv(RESULT_DIR + 'clean-' + savename + '.tsv', sep='\t')
    
    fig = visualize.word_frequency_chart(df)
    fig.savefig(RESULT_DIR + 'word-frequency-' + savename + '.png')
    # stopwordSet = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    
    reviews = []
    for phrase in df['Phrase']:
        words = word_tokenize(phrase)
        lemma_words = [lemmatizer.lemmatize(word) for word in words]
        reviews.append(lemma_words)
    
    if 'Sentiment' in df.columns:
        fig = visualize.sentiment_frequency_chart(df, sentiment_labels)
        fig.savefig(RESULT_DIR + 'sentiment-frequency-' + savename + '.png')

        sentiments = df.Sentiment.values
        return reviews, sentiments
    
    return reviews

if __name__ == "__main__":
    reviews, sentiments = preprocess_dataframe('train.tsv')
    reviews = preprocess_dataframe('test.tsv')
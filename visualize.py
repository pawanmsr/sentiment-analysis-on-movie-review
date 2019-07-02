import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DIR = "sentiment-analysis-on-movie-reviews"
RESULT_DIR = "results/"

sentimentLabels = {0 : 'negative', 1 : 'somewhat negative', 2 : 'neutral', 3 : 'somewhat positive', 4 : 'positive'}

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)

def drawInsights():
    trainDF = pd.read_csv(DEFAULT_DIR + '/train.tsv', sep = '\t', index_col = 'PhraseId')
    trainDF['WordCount'] = [len(x.split()) for x in trainDF['Phrase'].tolist()]
    
    sentimentFrequency = trainDF['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    plt.title('Sentiment Frequency')
    ax.pie(sentimentFrequency, labels = [sentimentLabels[x] for x in sentimentFrequency.index], autopct = '%1.1f%%')
    fig.savefig(RESULT_DIR + 'sentimentFrequency.png')

    wordFrequency = trainDF['WordCount'].value_counts()
    fig, ax = plt.subplots()
    plt.title("Word Count Frequency")
    plt.xlabel('Word Count')
    plt.ylabel('Phrases')
    ax.bar(wordFrequency.index, wordFrequency)
    fig.savefig(RESULT_DIR + 'wordCountFrequency.png')

def exitCommand(flag = 0):
    if flag < 3:
        print("incorrect directory")

    if flag < 1:
        print("correct usage: python {} <dataset directory>".format(os.path.basename(__file__)))
    
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        if not os.path.isdir(DEFAULT_DIR):
            exitCommand(0)
    elif len(sys.argv) == 2:
        if not os.path.isdir(sys.argv[1]):
            exitCommand(2)
        else:
            DEFAULT_DIR = sys.argv[1]
    else:
        exitCommand(0)
    
    drawInsights()
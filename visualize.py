import matplotlib.pyplot as plt
import pandas as pd

def sentiment_frequency_chart(df, labelDict):
    fig, ax = plt.subplots()
    plt.title('Sentiment Frequency')
    if 'Sentiment' in df.columns:
        sentimentFrequency = df['Sentiment'].value_counts()
        ax.pie(sentimentFrequency, labels = [labelDict[x] for x in sentimentFrequency.index], autopct = '%1.1f%%')
    return fig

def word_frequency_chart(df):
    fig, ax = plt.subplots()
    plt.title("Word Count Frequency")
    plt.xlabel('Word Count')
    plt.ylabel('Phrases')
    df['WordCount'] = [len(x.split()) for x in df['Phrase'].tolist()]
    wordFrequency = df['WordCount'].value_counts()
    ax.bar(wordFrequency.index, wordFrequency)
    return fig
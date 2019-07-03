from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import os
import argparse

from utils import preprocess_dataframe, submit
from utils import json_save, pickle_save, pickle_load
from glove import loadWordVectors
from models import build_cnn, build_dense_nn, build_lstm
from visualize import training_evolution_graph

import os

RESULT_DIR = "results/"
MODEL_DIR = 'models/'

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

NUM_CLASSES = 5
SEQUENCE_LENGTH = 60

def get_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', help = 'train model (default: dense model)', dest = 'train', default = False)
    group.add_argument('--test', help = 'test model (default: dense model)', dest = 'test', default = False)
    return parser.parse_args()

def train(model_type = 'DenseNN'):
    reviews, sentiments = preprocess_dataframe('train.tsv')
    y = to_categorical(sentiments)
    
    tokenize = Tokenizer()
    tokenize.fit_on_texts(reviews)
    pickle_save(tokenize, MODEL_DIR + 'tokenizer.pickle')

    X = pad_sequences(tokenize.texts_to_sequences(reviews), SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y)
    
    json_save(tokenize.word_index, 'word_index.json')
    word_vectors = loadWordVectors(tokenize.word_index)
    
    if model_type == 'LSTM':
        print('using LSTM based model')
        model = build_lstm(word_vectors = word_vectors, train_embeddings = True,
            sequence_length = SEQUENCE_LENGTH, num_classes = NUM_CLASSES)
    elif model_type == 'CNN':
        print('using CNN based model')
        model = build_lstm(word_vectors = word_vectors, train_embeddings = True,
            sequence_length = SEQUENCE_LENGTH, num_classes = NUM_CLASSES)
    else:
        print('using DenseNN model')
        model = build_lstm(word_vectors = word_vectors, train_embeddings = True,
            sequence_length = SEQUENCE_LENGTH, num_classes = NUM_CLASSES)
    
    print('commencing training')
    early_stopping = EarlyStopping(min_delta = 0.0001, mode = 'max',
        monitor = 'val_acc', patience = 2)
    history = model.fit(X_train, y_train, validation_data = (X_val, y_val),
        batch_size = 128, epochs = 10, callbacks = [early_stopping])
    fig = training_evolution_graph(history)
    fig.savefig(RESULT_DIR + model_type + '_training_evolution.png')
    model.save(MODEL_DIR + model_type + '.h5')
    print('training complete')

def test(model_type = 'DenseNN'):
    tokenize = pickle_load('tokenizer.pickle')
    reviews = preprocess_dataframe('test.tsv')
    X = pad_sequences(tokenize.texts_to_sequences(reviews), SEQUENCE_LENGTH)
    
    print('testing . . . ')
    model = load_model(MODEL_DIR + model_type + '.h5')
    pred = model.predict_classes(X)
    print('testing complete')
    submit(pred, model_type)

if __name__ == "__main__":
    args = get_arguments()
    if args.train:
        train(args.train)
    elif args.test:
        if args.test not in ['DenseNN', 'LSTM', 'CNN']:
            args.test = 'DenseNN'
        if not os.path.isfile(MODEL_DIR + args.test + '.h5'):
            print('pretrained model not found')
            train(args.test)
        test(args.test)
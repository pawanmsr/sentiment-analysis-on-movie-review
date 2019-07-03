from keras.layers import Dense, Flatten, Embedding, Conv2D, LSTM
from keras.models import Sequential

def build_dense_nn(word_vectors, train_embeddings = False, sequence_length = 60, num_classes = 5):
    model = Sequential()
    model.add(Embedding(word_vectors.shape[0], word_vectors.shape[1],
        input_length = sequence_length,
        weights = [word_vectors], trainable = train_embeddings))
    model.add(Flatten())
    model.add(Dense((word_vectors.shape[1] + num_classes)/4, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model

def build_cnn():
    pass

def build_lstm(word_vectors, train_embeddings = False, sequence_length = 60, num_classes = 5):
    model = Sequential()
    model.add(Embedding(word_vectors.shape[0], word_vectors.shape[1],
        input_length = sequence_length,
        weights = [word_vectors], trainable = train_embeddings))
    model.add(LSTM(128, dropout = 0.1, recurrent_dropout = 0.1, return_sequences = True))
    model.add(LSTM(32, dropout = 0.1, recurrent_dropout = 0.1))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model
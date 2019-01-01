from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

max_features = 10000
maxlen = 400
batch_size = 50

# load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# split test and validation sets
val_len = int(len(y_test)*0.15)
x_validation, y_validation = x_test[:val_len,:], y_test[:val_len]
x_test, y_test = x_test[val_len:, :], y_test[val_len:]

# create LSTM model with 512 units and dropout
model = Sequential()
model.add(Embedding(max_features, 50, input_length=maxlen, batch_input_shape=(batch_size, maxlen,)))
model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2, stateful=True))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# train model
model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_validation, y_validation))

#save model
model.save('my_model.h5')

# print model's score and accuracy
print('Score: %f\n Accuracy: %f' % model.evaluate(x_test, y_test, batch_size=batch_size))
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

max_features = 10000
maxlen = 400

(x_train, _), (x_test, _) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

encoding_dim = 80

input_seq = Input(shape=(None,maxlen))
encoded = Dense(encoding_dim, activation='relu')(input_seq)
decoded = Dense(maxlen, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=500, batch_size=128)
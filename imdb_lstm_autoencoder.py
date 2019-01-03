from numpy import array
from keras.models import Sequential
from keras.layers import TimeDistributed, Embedding, RepeatVector, Dense, LSTM
from keras.utils import plot_model
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 9
batch_size = 128

(x_train, _), (x_test, _) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# reshape input into [samples, timesteps, features]
timesteps = x_train.shape[1]
samples = x_train.shape[0]
x_train = x_train.reshape((samples, timesteps, 1))

model = Sequential()
# model.add(Embedding(max_features, 50, batch_input_shape=(batch_size, timesteps,1)))
model.add(LSTM(512, activation='relu'))
model.add(RepeatVector(timesteps))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train, epochs=300)
print(model.evaluate(x_test, x_test))

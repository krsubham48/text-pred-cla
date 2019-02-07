#import dependencies
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

#load data
data = open('alice.txt').read()
data = data.lower()

char_list = list(set(data))
char = sorted(char_list)

#dictionary of unique characters and their corresponding int
char_to_int = dict((c, i) for i, c in enumerate(char))
n_chars = len(data)
n_vocab = len(char)
print('Total characters: ', n_chars)
print('Total Vocab', n_vocab)

#creating a labeled dataset
seq_length = 200
dataX = []
dataY = []
for i in range(0, n_chars-seq_length, 1):
    seq_in = data[i:i+seq_length]
    seq_out = data[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print('Total Patterns: ', n_patterns)

print('The first 10 values of the 0th entry in dataX: ', dataX[0][:10])
print('The corresponding value of the 0th entry in dataY: ', dataY[0])

#reshape to match the layer dimensions
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab)
Y = np_utils.to_categorical(dataY)

print(Y.shape)
print(Y[0])

#create LSTM model
model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#train model
model.fit(X, Y, epochs=20, batch_size=128)

'''
now use model.predict(x) to test the working of model after training completes
'''
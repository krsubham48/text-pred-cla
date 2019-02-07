import numpy as np
import pandas as pandas
from bs4 import BeautifulSoup

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

data = open('taletwocities.txt').read().lower()

def clean_review(raw_input):
	no_markup = BeautifulSoup(raw_input, 'html5lib').get_text()
	words = no_markup.split()
	return words
review_clean = clean_review(data)
word = sorted(list(set(review_clean)))
char_to_int = dict((c, i) for i, c in enumerate(word))

n_chars = len(review_clean)
n_vocab = len(word)
print('Total characters: ', n_chars)
print('Total Vocab', n_vocab)

seq_length = 3
dataX = []
dataY = []
for i in range(0, n_chars-seq_length, 1):
    seq_in = review_clean[i:i+seq_length]
    seq_out = review_clean[i+seq_length]
    dataX.append([char_to_int[word] for word in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print('Total Patterns: ', n_patterns)

X = np.reshape(dataX, (-1, seq_length, 1))
X = X/float(n_vocab)
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, Y, epochs=20, batch_size=128)
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D,Bidirectional,LSTM
from keras.datasets import imdb
from sklearn.metrics import accuracy_score

max_features = 6000
max_length = 400

import numpy as np
old = np.load
np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
np.load = old
del(old)
print(x_train[0])
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

xInput = Input(batch_shape=(None, max_length))
xEmbed = Embedding(max_features, 60, input_length = max_length)(xInput)
xLstm = Bidirectional(LSTM(60))(xEmbed)
xOutput = Dense(1, activation='sigmoid')(xLstm)
model = Model(xInput, xOutput)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()
model.fit(x_train, y_train, batch_size=32, epochs = 1)
y_hat = model.predict(x_test, batch_size=32)
y_hat_class = np.round(y_hat, 0)
y_hat_class.shape = y_test.shape

print (("Test accuracy:"),(np.round(accuracy_score(y_test, y_hat_class), 3)))

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM

x = np.array([[[1,2],[2,3],[3,4],[4,5]]])
y = np.array([[[3],[4],[5],[6]]])

model = Sequential()

model.add(LSTM(2, return_sequences=True,input_shape=(4,2)))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer='adam')
model.summary()

model.fit(x,y,batch_size = 1)
predy = model.predict((x))
print(predy)



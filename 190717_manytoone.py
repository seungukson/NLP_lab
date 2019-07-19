import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM

x = np.array([[[i for i in range(j-10, j)] for j in range(10, 50)]])
y = np.array([[i] for i in range(10, 50)])
model = Sequential()
x = x.reshape(-1,10,1)
print(x.shape, y.shape)
model.add(Bidirectional(LSTM(2, return_sequences=False), input_shape=(10, 1)))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer='adam')
model.summary()


model.fit(x, y, epochs = 50, batch_size = 1)
testx = np.array([[[i] for i in range(50, 60)]])
predy = model.predict(x)
print(predy)



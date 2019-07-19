import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional
import matplotlib.pyplot as plt

a = np.sin(2* np.pi * 0.03 * np.arange(0,100)+ np.random.random(100))
# np.random.random(100) : 0.0-~
def generateX(a,n):
    x_train = [];
    y_train = []
    for i in range(len(a)):
        x = a[i:(i+n)]
        if len(x) >= n and (i+n) <len(a):
            x_train.append(x)
            y_train.append(a[i+n])
    return np.array(x_train), np.array(y_train)

x, y = generateX(a,10)
x = x.reshape(-1,10,1)
y = y.reshape(-1,1)

x_train = x[:70, : , :]
y_train = y[:70, :]
x_test = x[70:, : , :]
y_test = y[70: , :]
plt.plot(x_train[:, 9, :], 'o-')

model = Sequential()
model.add(LSTM(2, input_shape = (10,1), return_sequences=True))
model.add(Bidirectional(LSTM(5)))
model.add(Dense(1))
model.compile(loss='mse', optimizer ='adam')
model.summary()

model.fit(x_train, y_train, epochs = 300, batch_size = 20, verbose = 1)
y_hat = model.predict(x_test, batch_size = 1)
a_axis = np.arange(0,len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))
plt.plot(a_axis, y_train.reshape(70,),'o-')
plt.plot(b_axis, y_hat.reshape(20,), 'o-')
plt.show()
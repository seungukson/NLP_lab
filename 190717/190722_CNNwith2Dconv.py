from keras.models import Model
from keras.layers import Dense, LSTM, Convolution2D, MaxPooling2D,Input,Flatten
import numpy as np
from keras import backend as K
x_train = np.random.random([10,8,5,1])# data갯수,가로8,세로5,채널수
y_train = np.random.choice([0,1],[10,1])
x_test = np.random.random([2,8,5,1])
K.clear_session();
xinput = Input(batch_shape=(None, 8, 5, 1))
conv2 = Convolution2D(12, 3, activation='relu')(xinput)
pool2 = MaxPooling2D((2,2),strides=(1,1))(conv2)
flat = Flatten()(pool2)
hidden1 = Dense(20,activation='relu')(flat)
youtput = Dense(1,activation='sigmoid')(hidden1)

model = Model(xinput,youtput)
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs = 30,batch_size=3)
model.summary()


print(model.predict(x_test))



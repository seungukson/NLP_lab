from keras.models import Model
from keras.layers import Dense, Input, LSTM,Bidirectional
# from keras import backend as K
import numpy as np
from keras.utils import to_categorical

x_train = np.random.random([10, 8, 5])
x_train = np.round(x_train, 3)
print(x_train)

y_train = np.random.randint(0, 5, (10,8))
z_test = np.random.random([1,8,5])
print(y_train)
y_arr = np.zeros((10, 5))
for i, v in enumerate(y_train):
    y_arr[i][v] = 1
# K.clear_session()
y_cat = to_categorical(y_train)
xInput = Input(batch_shape=(None, 8, 5))
Hidden = Bidirectional(LSTM(20,return_sequences=True))(xInput)
Hidden2 = Bidirectional(LSTM(10, return_sequences=True))(Hidden)
yOutput = Dense(5,activation='softmax')(Hidden2)
# 중간레이어 알아보기
# h = Model(x_input, hidden)
model = Model(xInput, yOutput)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_cat, epochs=50,batch_size=2,verbose=1)

predY = model.predict(z_test,batch_size=1)

print(np.argmax(predY))
# print(np.argmax(predY))
print(z_test)
print(model.summary())
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
import numpy as np
from keras.utils import to_categorical

x_train = np.random.random([10, 8])
x_train = np.round(x_train, 3)
print(x_train)

y_train = np.random.randint(0, 5, (10,1))
z_test = np.random.random([1,8])
print(y_train)
y_arr = np.zeros((10, 5))
for i, v in enumerate(y_train):
    y_arr[i][v] = 1

git_branch_test = "It's my test script"

y_cat = to_categorical(y_train)
xInput = Input(batch_shape=(None, 8))
Hidden = Dense(50,activation='relu')(xInput)
yOutput = Dense(5,activation='softmax')(Hidden)

model = Model(xInput, yOutput)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_cat, epochs=50,batch_size=1,verbose=1)

# predYtest = model.predict(z_test,batch_size=1)
#
# print(predYtest,np.argmax(predYtest))
# print(model.summary())


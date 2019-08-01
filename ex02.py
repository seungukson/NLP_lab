# 8단어, 1단어는 3개의feature, 10개의 리뷰.
# CNN통과 후 4*12의 데이터를 RNN을 통과시켜서 FFN

from keras.models import Model
from keras.layers import Input, LSTM, Convolution1D, MaxPooling1D,Dense
import numpy as np

x_train = np.random.random([10, 8, 3])
y_train = np.random.randint(0, 1, (10, 1))

x_test= np.random.random([4,8,3])

xinput = Input(batch_shape=(None, 8, 3))
conv = Convolution1D(12, 3, activation='relu')(xinput)
pool = MaxPooling1D(3)(conv)
rn = LSTM(10)(pool)
ffn = Dense(50, activation='relu')(rn)
youtput = Dense(1, activation='sigmoid')(ffn)
model = Model(xinput, youtput)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1)

predY = model.predict(x_test)
print(predY)
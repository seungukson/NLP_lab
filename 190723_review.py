text = ['dog like cat', 'dog dog fish river',
        'cat dog humnan', 'cat cute odd eye', 'dog bark cut cry',
        'cat run dog run', 'cat cat cat dog dog dog', 'dog eat cat cat dog']

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Model
from keras.layers import Convolution1D, GlobalMaxPooling1D, Dense, Input
from keras import backend as K
import numpy as np

vectorize = TfidfVectorizer()

y_train = [1 if i > 0.5 else 0 for i in np.random.random([8,1])]
print(y_train)
y_train = np.array(y_train).reshape(-1,1)
vc = vectorize.fit(text)
print(vc, type(vc))
tr_x = vc.transform(text).toarray()
# print(tr.toarray())
# print(np.toarray(tr))
tr_x = np.array(tr_x).reshape(-1,14,1)
# print(tr_x.shape)
K.clear_session()
xInput = Input(batch_shape=(None, 14, 1))
xConv = Convolution1D(10, 3, activation='relu')(xInput)
xPool = GlobalMaxPooling1D()(xConv)
xFFN = Dense(10, activation='relu')(xPool)
xOutput = Dense(1, activation='sigmoid')(xFFN)

model = Model(xInput, xOutput)
model.compile(loss='binary_crossentropy',optimizer='adam')
model.summary()

x_test = np.array([i for i in np.random.random([3,14,1])])

model.fit(tr_x, y_train,epochs=20,batch_size=5)

predY = model.predict(x_test)
print(predY, np.argmax(predY))
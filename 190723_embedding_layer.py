text = ['dog like cat', 'dog dog fish river',
        'cat dog humnan', 'cat cute odd eye', 'dog bark cut cry',
        'cat run dog run', 'cat cat cat dog dog dog', 'dog eat cat cat dog']
# dictionary 만들기
from keras_preprocessing import sequence
total = set([i for t in text for i in t.split(" ")])
word_dict = {w:i+1 for i, w in enumerate(set([i for t in text for i in t.split(" ")]))}
tm = [[word_dict[a] for a in t.split(" ")] for t in text]
# padding
import numpy as np

maxLen = np.max([len(s) for s in tm])
for i in range(len(tm)):
    tm[i] = [0]*(maxLen - len(tm[i]))+tm[i]
print(tm)

# word Encoding : 그냥 단순히 수치화
# 단어나 문장의 의미는 전혀고려하지 않았던 인코딩.
# word Embedding : 수치데이터가 의미를 갖도록한다.
from keras.models import Model
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Flatten, Dense,GlobalMaxPooling1D,Conv1D
from keras import backend as K
x_train = sequence.pad_sequences(tm, maxlen=6)
y_train = np.random.randint( 0, 2, [8, 1] )
# print(y_train)
K.clear_session()

xInput = Input(batch_shape=(None, maxLen))
xEmbed = Embedding(input_dim=15, output_dim=4, input_length=6)(xInput)
xConv = Convolution1D(8, 2, activation='relu')(xEmbed)
xPool = MaxPooling1D(2, strides=1)(xConv)
xFlat = Flatten()(xPool)
xHidden = Dense(15, activation='relu')(xFlat)
xOutput = Dense(1, activation='sigmoid')(xHidden)

model = Model(xInput, xOutput)
model.summary()
model.compile(loss='binary_crossentropy',  optimizer='adam')
model.fit(x_train, y_train, epochs=10,batch_size=4)


# 일반 문자열을 1*6의 숫자벡터로 변환해주는 함수
def encode(text):
    text = [word_dict[i] for i in text.split(' ')]
    text = [0]*(maxLen-len(text))+text
    return np.array([text])
tt = 'dog cat dog eat'
z_test = encode(tt)
z_test
maxLen
model2 = Model(xInput, xEmbed)
pred = model2.predict(z_test)
print(pred)
pred2 = model.predict(z_test)
print(pred2)


#단어를 수치화하는 방법에 대해알아보자.
#Encoding과 Embedding이 있다.
#Encoding은 단어사용 빈도 등에 의한 단순 수치화=TF-IDF, CountVector
#Embedding은 단어의 의미를 살린 수치화: CBOW,Skip-gram
text = ['dog like cat', 'dog dog fish river',
        'cat dog humnan', 'cat cute odd eye', 'dog bark cut cry',
        'cat run dog run', 'cat cat cat dog dog dog', 'dog eat cat cat dog']
# vocabulary: cat dog like fish river human cute odd eye bark cut cry run eat
#vocab = ['cat', 'dog', 'like', 'fish', 'river', 'human', 'cute', 'odd', 'eye', 'bark', 'cut', 'cry', 'run', 'eat']
# count =
from keras.models import Model
from keras.layers import Dense, LSTM, Convolution1D, GlobalMaxPooling1D, Input
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
# import tensorflow as tf
cntvec = CountVectorizer()
ct = cntvec.fit(text)
# print(ct.vocabulary_) # 총 단어의 갯수를 셈
ct_vec = ct.transform(text).toarray()
# print(ct_vec) #원소별 수치화

tfvec = TfidfVectorizer()
tv =tfvec.fit(text)

tv_vec = np.round(tv.transform(text).toarray(),3)
# print(tv_vec)
# print(tv.transform(text))
# print(tv)

## 학습시켜보기
y_train = np.random.choice([0, 1], [10])
x_train = tv_vec
print(x_train, y_train)

xInput = Input(batch_shape=(None, ))
xConv = Convolution1D(10, 3)(xInput)
xPool = GlobalMaxPooling1D()(xConv)


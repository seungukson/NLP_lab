from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import string

#이상한 나라의 엘리스 소설을 읽어온다.

with open('alice_in_wonderland.txt', 'r') as content_file:
    content = content_file.read()
content
content2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in content]).split())

tokens = nltk.word_tokenize(content2)
tokens = [word.lower() for word in tokens if len(word) >= 2]
tokens
#Trigram list

N = 3
quads = list(nltk.ngrams(tokens, N))

quads[:3]

newl_app = []
for ln in quads:
    newl = " ".join(ln)
    newl_app.append(newl)

newl_app[:3]

#Vectorizing the words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

x_trigm = []
y_trigm = []

for l in newl_app:
    x_str = " ".join(l.split()[0:N-1])  # trigram 중 앞 부분 2 단어.
    y_str = l.split()[N-1]              # trigram 중 마지막 1 단어.
    x_trigm.append(x_str)
    y_trigm.append(y_str)

x_trigm
y_trigm
x_trigm_check = vectorizer.fit_transform(x_trigm).todense()
y_trigm_check = vectorizer.fit_transform(y_trigm).todense()
x_trigm_check[0][0]
#Dictionaries from word to integer and integer to word

dictnry = vectorizer.vocabulary_
rev_dictnry = {v:k for k,v in dictnry.items()}

X = np.array(x_trigm_check)
Y = np.array(y_trigm_check)

Xtrain, Xtest, Ytrain, Ytest,xtrain_tg,xtest_tg = train_test_split(X, Y,x_trigm, test_size=0.3,random_state=42)

print("X Train shape",Xtrain.shape, "Y Train shape" , Ytrain.shape)
print("X Test shape",Xtest.shape, "Y Test shape" , Ytest.shape)

#Model Building

from keras.layers import Input, Dense, Dropout
from keras.models import Model

np.random.seed(42)

BATCH_SIZE = 128
NUM_EPOCHS = 10

input_layer = Input(shape = (Xtrain.shape[1],))
first_layer = Dense(1000, activation='relu')(input_layer)
first_dropout = Dropout(rate = 0.5)(first_layer)

second_layer = Dense(800, activation='relu')(first_dropout)

third_layer = Dense(1000, activation='relu')(second_layer)
third_dropout = Dropout(rate = 0.5)(third_layer)

fourth_layer = Dense(Ytrain.shape[1], activation='softmax',name = "fourth")(third_dropout)


history = Model(input_layer, fourth_layer)
history.compile(optimizer = "adam",loss="categorical_crossentropy", metrics=["accuracy"])

print (history.summary())

#Model Training
history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1,validation_split = 0.2)

#Model Prediction

Y_pred = history.predict(Xtest)

#Sample check on Test data

print ("Prior bigram words","| Actual","| Predicted","\n")

for i in range(10):
    print (i,xtest_tg[i], "|", rev_dictnry[np.argmax(Ytest[i])], "| ", end='')
    idx = np.flipud(Y_pred[i].argsort())[:5]
    for k in idx:
        print(rev_dictnry[k], ' ', end='')
    print()

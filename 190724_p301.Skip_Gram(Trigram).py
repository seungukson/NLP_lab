from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import nltk
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from textUtils import textPreprocessing

lines = []
fin = open("190724_datasets/alice_in_wonderland.txt", "r")
   
for line in fin:
    if len(line) <= 1:
        continue
    lines.append(textPreprocessing(line))
fin.close()

counter = collections.Counter()
for line in lines:
    for word in nltk.word_tokenize(line):
        counter[word.lower()] += 1

word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
idx2word = {v:k for k,v in word2idx.items()}

xs = []
ys = []

for line in lines:
    embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)] 
    triples = list(nltk.trigrams(embedding))
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    xs.extend(w_centers)
    ys.extend(w_lefts)
    xs.extend(w_centers)
    ys.extend(w_rights)

          
print (len(word2idx))
vocab_size = len(word2idx)+1
ohe = OneHotEncoder(categories = [range(vocab_size)])

X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()

Xtrain, Xtest, Ytrain, Ytest, xstr, xsts = train_test_split(X, Y, xs, test_size=0.3, random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

np.random.seed(42)

BATCH_SIZE = 128
NUM_EPOCHS = 20

input_layer = Input(shape = (Xtrain.shape[1],), name="input")
first_layer = Dense(300, activation='relu', name = "first")(input_layer)
first_dropout = Dropout(0.5, name="firstdout")(first_layer)
second_layer = Dense(2, activation='relu', name="second")(first_dropout)
third_layer = Dense(300, activation='relu', name="third")(second_layer)
third_dropout = Dropout(0.5, name="thirdout")(third_layer)
fourth_layer = Dense(Ytrain.shape[1], activation='softmax', name = "fourth")(third_dropout)

history = Model(input_layer, fourth_layer)
history.compile(optimizer = "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_split = 0.2)

encoder = Model(history.input, history.get_layer("second").output)
reduced_X = encoder.predict(Xtest)

x = encoder.predict(Xtest[0])
y = encoder.predict(Xtest[1])
print(np.argmax(Xtest[0]))
np.sqrt(np.sum((x-y)**2))


# final_pdframe = pd.DataFrame(reduced_X)
# final_pdframe.columns = ["xaxis","yaxis"]
# final_pdframe["word_indx"] = xsts
# final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word)
# final_pdframe.head()
#
# vis_df = final_pdframe.sample(100)

# labels = list(vis_df["word"])
# xvals = list(vis_df["xaxis"])
# yvals = list(vis_df["yaxis"])

# #in inches
# plt.figure(figsize=(14, 10))
#
# for i, label in enumerate(labels):
#     x = xvals[i]
#     y = yvals[i]
#
#     plt.scatter(x, y)
#     plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points', ha='right',va='bottom')
#
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.show()



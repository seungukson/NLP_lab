import collections
import itertools
import nltk
import numpy as np
import matplotlib.pyplot as plt
import os
import random


# 1 Mary moved to the bathroom.
# 2 John went to the hallway.
# 3 Where is Mary? (tab)bathroom(tab)1

def get_data(infile):
    stories, questions, answers = [], [], []
    story_text = []
    fin = open(infile, "r")
    for line in fin:
        line = line.strip()                # strip() : 양쪽 끝에 있는 공백, \t, \n 제거
        lno, text = line.split(" ", 1)     # 맨 앞의 라인 번호 분리
        if "\t" in text:                   # 세 번째 문장에는 \t가 두개 있음.
            question, answer, _ = text.split("\t")
            stories.append(story_text)     # 처음 두 문장
            questions.append(question)     # 세 번째 문장의 질문
            answers.append(answer)         # 세 번째 문장의 답변
            story_text = []
        else:
            story_text.append(text)        # 처음 두 문장이 들어감
    fin.close()
    return stories, questions, answers

# get the data
data_train = get_data("C:/Users/ssson/PycharmProjects/NLP_lab/190801_/qa1_single-supporting-fact_train.txt")
data_test = get_data("C:/Users/ssson/PycharmProjects/NLP_lab/190801_/qa1_single-supporting-fact_test.txt")

print("Train observations:",len(data_train[0]),"Test observations:", len(data_test[0]))

print("스토리 = ", data_train[0][0])
print("질  문 = ", data_train[1][0])
print("답  변 = ", data_train[2][0])

dictnry = collections.Counter()
type(dictnry)
dictnry
data_train
data_test
for stories, questions, answers in [data_train, data_test]:
    for story in stories:
        for sent in story:
            for word in nltk.word_tokenize(sent):
                dictnry[word.lower()] += 1

    for question in questions:
        for word in nltk.word_tokenize(question):
            dictnry[word.lower()] += 1

    for answer in answers:
        for word in nltk.word_tokenize(answer):
            dictnry[word.lower()] += 1

dictnry.most_common()
word2indx = {w: (i + 1) for i, (w, _) in enumerate(dictnry.most_common())}
word2indx
word2indx["PAD"] = 0
indx2word = {v: k for k, v in word2indx.items()}
indx2word
vocab_size = len(word2indx)
vocab_size
print("vocabulary size:", len(word2indx))
print(word2indx)

story_maxlen = 0
question_maxlen = 0

for stories, questions, answers in [data_train, data_test]:
    for story in stories:
        story_len = 0
        for sent in story:
            swords = nltk.word_tokenize(sent)
            story_len += len(swords)
        if story_len > story_maxlen:
            story_maxlen = story_len

    for question in questions:
        question_len = len(nltk.word_tokenize(question))
        if question_len > question_maxlen:
            question_maxlen = question_len

print("Story maximum length:", story_maxlen, "Question maximum length:", question_maxlen)

from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

# Converting data into Vectorized form
# Xstrain[0] = [0,0,8,21,1,2,12,3,9,7,1,2,13,3] - 14개
# Xqtrain[0] = [4,5,8,6] - 4개
# Ytrain[0] = [0,0,0,0,...0,1,0,0,0,] - 22개의 one-hot
xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
xs
stories
word2indx
# list of list 연습
for story, question, answer in zip(stories, questions, answers):
    print(story)
    print(xs)
    xs = list(itertools.chain.from_iterable(xs))
    print(xs)
    break


def data_vectorization(data, word2indx, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
        xs = list(itertools.chain.from_iterable(xs))  # 2개 스토리를 하나로 합친다
        xq = [word2indx[w.lower()] for w in nltk.word_tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2indx[answer.lower()])

    return pad_sequences(Xs, maxlen=story_maxlen), \
           pad_sequences(Xq, maxlen=question_maxlen), \
           np_utils.to_categorical(Y, num_classes=len(word2indx))


Xstrain, Xqtrain, Ytrain = data_vectorization(data_train, word2indx, story_maxlen, question_maxlen)
Xstest, Xqtest, Ytest = data_vectorization(data_test, word2indx, story_maxlen, question_maxlen)

print("Train story", Xstrain.shape, "Train question", Xqtrain.shape, "Train answer", Ytrain.shape)
print("Test story", Xstest.shape, "Test question", Xqtest.shape, "Test answer", Ytest.shape)

Xstrain[0]
#### Model Parameters

EMBEDDING_SIZE = 128
LATENT_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 40

# Inputs
story_input = Input(shape=(story_maxlen,))
question_input = Input(shape=(question_maxlen,))

# Story encoder embedding
story_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE,
                         input_length=story_maxlen)(story_input)
story_encoder = Dropout(rate = 0.8)(story_encoder)

# Question encoder embedding
question_encoder = Embedding(input_dim=vocab_size,output_dim=EMBEDDING_SIZE,
                            input_length=question_maxlen)(question_input)
question_encoder = Dropout(rate = 0.7)(question_encoder)

# Match between story and question
match = dot([story_encoder, question_encoder], axes=[2,2])

# Encode story into vector space of question
story_encoder_c = Embedding(input_dim=vocab_size,output_dim=question_maxlen,
                           input_length=story_maxlen)(story_input)
story_encoder_c = Dropout(rate = 0.7)(story_encoder_c)

# Combine match and story vectors
response = add([match, story_encoder_c])
response = Permute((2, 1))(response)

# Combine response and question vectors to answers space
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(rate = 0.8)(answer)
answer = Dense(vocab_size)(answer)
output = Activation("softmax")(answer)

model = Model(inputs=[story_input, question_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
print (model.summary())

#### Model Training

history = model.fit([Xstrain, Xqtrain], [Ytrain], batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS, validation_data=([Xstest, Xqtest], [Ytest]))

#### plot accuracy and loss plot
plt.title("Episodic Memory Q & A Accuracy")
plt.plot(history.history["acc"], color="g", label="train")
plt.plot(history.history["val_acc"], color="r", label="validation")
plt.legend(loc="best")
plt.show()

#### get predictions of labels
from sklearn.metrics import accuracy_score
ytest = np.argmax(Ytest, axis=1)
Ytest_ = model.predict([Xstest, Xqtest])
ytest_ = np.argmax(Ytest_, axis=1)
print("score:", accuracy_score(ytest, ytest_))
""
#### Select Random questions and predict answers
NUM_DISPLAY = 10

for i in random.sample(range(Xstest.shape[0]), NUM_DISPLAY):
    story = " ".join([indx2word[x] for x in Xstest[i].tolist() if x != 0])
    question = " ".join([indx2word[x] for x in Xqtest[i].tolist()])
    label = indx2word[ytest[i]]
    prediction = indx2word[ytest_[i]]
    print(story, question, label, prediction)

story
question
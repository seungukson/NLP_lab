from keras.datasets import imdb
import numpy as np

old = np.load
np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=6000)
np.load = old
del(old)

print(x_train[0])
print(y_train[0])

wind = imdb.get_word_index()
wind['kagan']
revind = dict((v, k) for k, v in wind.items())

def decode(sent_list):
    new_words = []
    for i in sent_list:
        new_words.append(revind.get(i-3, '*'))
    comb_words = ' '.join(new_words)
    return comb_words

decode(x_train[0])

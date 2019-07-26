import gensim
import numpy as np

path = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

model['happy']

def uclidDist(word1, word2):
    return np.sqrt(np.sum((model[word1]-model[word2])**2))
def cosDist(word1, word2):
    x = model[word1]
    y = model[word2]
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def mingDist(word1, word2,p):
    return np.sum((model[word1]-model[word2])**p)**(1/p)

print(uclidDist('mother', 'father'), uclidDist('mother','son'),uclidDist('mother','woman'));
print(cosDist('mother', 'father'), cosDist('mother','son'),cosDist('mother','woman'));
print(mingDist('mother', 'father',4), mingDist('mother','son',4),mingDist('mother','woman',4));




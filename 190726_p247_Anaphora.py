import nltk
from nltk.chunk import tree2conlltags
from nltk.corpus import names
import random

def feature(word):
    return {'last(1)' : word[-1]}

males = [(name, 'male') for name in names.words('male.txt')]
females = [(name, 'female') for name in names.words('female.txt')]
combined = males + females
random.shuffle(combined)
training = [(feature(name), gender) for (name, gender) in combined]

males[:10]

females[:10]

training[:10]

classifier = nltk.NaiveBayesClassifier.train(training)

sentences = [
    "John is a man. He walks",
    "John and Mary are married. They have two kids",
    "In order for Ravi to be successful, he should follow John",
    "John met Mary in Barista. She asked him to order a Pizza"
]
def gender(word):
    return classifier.classify(feature(word))

for sent in sentences:
    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False)
    stack = []
    print(sent)
    items = tree2conlltags(chunks)#iob tagging
    for item in items:
        if item[1] == 'NNP' and (item[2] == 'B-PERSON' or item[2] == 'O'):
            stack.append((item[0], gender(item[0])))
        elif item[1] == 'CC':
            stack.append(item[0])
        elif item[1] == 'PRP':
            stack.append(item[0])
    print("\t {}".format(stack))

items

print(chunks)


##주피터노트북 참조
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
#from sklearn.datasets import fetch_20newsgroups

#newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

#with open('./data/news.data', 'wb') as f:
#    pickle.dump(newsData , f, pickle.HIGHEST_PROTOCOL)

with open('./news.data', 'rb') as f:
    newsData = pickle.load(f)

f = open('a.txt')

news = newsData.data
print(len(news))
print(news[0])


print(newsData.target_names)
print(len(newsData.target_names))

news1 = []
for doc in news:
    news1.append(re.sub("[^a-zA-Z]", " ", doc))
#영문자 이외의 문자는바꾼다.

stop_words = stopwords.words('english')
news2 = []
for doc in news1:
    doc1 = []
    for w in doc.split():
        w = w.lower()
        if len(w) > 3 and w not in stop_words:
            doc1.append(w)
    news2.append(' '.join(doc1))

print(news2[0])
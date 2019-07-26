
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import  cosine_similarity
import numpy as np
statements = ['ruled india',
              'Chalukyas ruled Badami',
              'So manykingdoms ruled India',
              'Lalbagh is a botanical garden in India']
vec = TfidfVectorizer()
tfIdf = vec.fit_transform(statements).toarray()
#fit: voca만들기.
#transform: 만들어주기
#todense: matrix - 2차원만지원, toarray: array로 변환-다차원지원
tfIdf[3]

def cos_distance(x, y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

print('문서(1) - (3) : ', cos_distance(tfIdf[0],tfIdf[2]))
print('문서[3) - (4) : ', cos_distance(tfIdf[2], tfIdf[3]))

print(cosine_similarity([tfIdf[0]], [tfIdf[2]]),
cosine_similarity([tfIdf[2]], [tfIdf[3]]))

# 문사의 유사도측정에서 tf-idf는 심플하게 구현해볼수있으므로 외워둘것.


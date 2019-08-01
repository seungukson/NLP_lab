import numpy as np
x = np.array([1,1,1,2])
y = np.array([1,1,1,2])

def cosDist(a, b):
    return np.dot(a, b)/ (np.linalg.norm(x)*np.linalg.norm(y))

print(np.dot(x,y))
print(cosDist(x,y))

#correlation구하기
np.corrcoef(x,y)

# dot의 또하나의 의미: 변환

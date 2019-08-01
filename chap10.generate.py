import numpy as np
r = np.array([0.7, 0.8])
m1 = r/r.sum()
m1

softmax = np.exp(r) / np.sum(np.exp(r))
print(m1, softmax)
# 47.5% <->5.25% 인데.... 더 격차를 벌릴순 없을까?
# e^xi/b... b값을 1이하로 내릴수록 격차가 커진다. 이것을 이용하자.

np.random.multinomial(100,[0.5,0.1,0.1,0.2,0.10,199],10)
#다항분포


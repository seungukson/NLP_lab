text = ['dog like cat', 'dog dog fish river',
        'cat dog humnan', 'cat cute odd eye', 'dog bark cut cry',
        'cat run dog run', 'cat cat cat dog dog dog', 'dog eat cat cat dog']



#dictionary 만들기

total = []
# wordlist = [y for y in t.split(" ") for t in text]
for t in text:
    for i in t.split(" "):
        total.append(i)
total = set(total)

total = set([i for t in text for i in t.split(" ")])

word_dict = {w:i+1 for i, w in enumerate(set([i for t in text for i in t.split(" ")]))}
# number = [i for i in range(total.len)]

# print(total)

print(word_dict)
text_vec = [[]]

# large_list= []
# for t in text:
#     tmp_list = []
#     for a in t.split(" "):
#         tmp_list.append(word_dict[a])
#     large_list.append(tmp_list)
# print(large_list)

tm = [[word_dict[a] for a in t.split(" ")] for t in text]

print(tm)
# padding
import numpy as np
max_len=0

pad_ary = np.zeros((np.array(tm).shape[0], 6))
print(pad_ary)
# for i in tm:
#     if(max_len<len(i)):
#         max_len=len(i)
# 좀 더 좋은 코드
maxLen = np.max([len(s) for s in tm])

# for i, v1 in enumerate(tm):
#     for j,v2 in enumerate(v1):
#         for k in range(max_len-len(tm[i])):
#             tm[i].insert(0, 0)
# print(tm)

for i, v1 in enumerate(tm):
    zero_num = maxLen - len(tm[i])
    print(i, v1, zero_num)
    for j, v2 in enumerate(v1):
        pad_ary[i][j+zero_num] = v2

for i in range(len(tm)):
    tm[i] = [0]*(maxLen - len(tm[i]))+tm[i]
# print(tm)
print(pad_ary)

# word Encoding : 그냥 단순히 수치화
# 단어나 문장의 의미는 전혀고려하지 않았던 인코딩.
# word Embedding : 수치데이터가 의미를 갖도록한다.
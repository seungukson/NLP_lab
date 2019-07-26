import nltk

sentence = 'I like you asdfasdf very much'
tok = nltk.word_tokenize(sentence)

pos = nltk.pos_tag(tok) # 토큰으로 변환한 것을 넣어줌.
pos #part of speech: pos

# nltk에 내장된 내장 태거.. 를 이용해 태깅
# ex : prp:대명사, vrp:동사, rb:부사/형용사


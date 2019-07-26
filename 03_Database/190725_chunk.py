import nltk
sentence = 'I like you very much in the morning'
sentences = nltk.sent_tokenize(sentence)
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print("word:", words)
    tags = nltk.pos_tag(words)
    print("tags:", tags)
    chunks = nltk.ne_chunk(tags)
    print("chunks:", chunks)

print(sentences)
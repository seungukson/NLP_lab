import nltk

def understandWordSenseExamples():
    words = ['wind', 'date', 'left']
    print("-- examples --")
    for word in words:
        syns = nltk.corpus.wordnet.synsets(word)
        for syn in syns[:2]:
            for example in syn.examples()[:2]:
                print("{} -> {} -> {}".format(word, syn.name(), example))

def understandBuiltinWSD():
    print("-- built-in wsd --")
    maps = [
        ('Is it the fish net that you are using to catch fish ?', 'fish', 'n'),
        ('Please dont point your finger at others.', 'point', 'n'),
        ('I went to the river bank to see the sun rise', 'bank', 'n'),
    ]
    for m in maps:
        print("Sense '{}' for '{}' -> '{}'".format(m[0], m[1], nltk.wsd.lesk(m[0], m[1], m[2])))

understandWordSenseExamples()

understandBuiltinWSD()

syns = nltk.corpus.wordnet.synsets('fish')
syns

nltk.corpus.wordnet.synset('pisces.n.02').lemma_names()

nltk.corpus.wordnet.synset('pisces.n.02').definition()

nltk.corpus.wordnet.synset('savings_bank.n.02').definition()


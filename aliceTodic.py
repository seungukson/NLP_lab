from textUtils import textPreprocessing

lines = []
fin = open("190724_datasets/alice_in_wonderland.txt", "r")

for line in fin:
    if len(line) <= 1:
        continue
    lines.append(textPreprocessing(line))
fin.close()

for line in lines:
    line.split(" ")
word_set = set([t for line in lines for t in line.split(" ")])
print(word_set)
len(word_set)

alice_dict ={ word: i for i, word in enumerate(word_set)}
print(alice_dict)
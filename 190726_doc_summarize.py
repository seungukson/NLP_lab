#문서의 주제를 요약한다.

from gensim.summarization import summarize
from bs4 import BeautifulSoup
import requests

url = 'http://scigen.csail.mit.edu/scicache/269/scimakelatex.25977.Admoni.Moskalskaia.Schendels.html'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
data = soup.get_text()
print(data)
pos1 = data.find('Introduction') + len("Introduction")
pos1

pos2 = data.find("Related Work")
pos2

text = data[pos1:pos2].strip()
summary = summarize(text, ratio=0.1)
print("PAPER URL: {}".format(url))
print("GENERATED SUMMARY: {}".format(summary))
print()

print(text)
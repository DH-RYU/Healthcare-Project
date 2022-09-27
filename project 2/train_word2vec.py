from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from tqdm.auto import tqdm
X_train = open('X_train.txt','r')

word_list = []

for line in tqdm(X_train.readlines()) :
    words = line.strip()
    words = words.split(',')[1:]
    word_list.append(words)

model = Word2Vec(sentences=word_list, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

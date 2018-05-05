# -*- coding: utf-8 -*-
from gensim.models import word2vec

#训练word2vec
sentences = word2vec.Text8Corpus("poetry_data_cut")
model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

print(model)

#保存模型
fname = "word_vec_model"
model.save(fname)


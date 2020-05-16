# coding: utf-8

from code.processData import read_corpus, cleanData, createVocab
from code.representText import Vectorizer
from code.similarSearch import Query
from code.correctText import spell_corrector


# 读取文件
qlist, alist = read_corpus(filename = '../data/train-v2.0.json')

# 处理文本
new_list = cleanData(qlist)

# 构造句子向量
vocab_count,count = createVocab(new_list)
vectorizer = Vectorizer()
X_tfidf = vectorizer.vectorizerTFIDFs(vocab_count, count, new_list)



while True:
    query = input('please ask: ')
    query = spell_corrector(query)
    query = cleanData(query)
    query_tfidf = vectorizer.vectorizerTFIDF(query, vocab_count, count, qlist)
    answer = Query(query_tfidf, X_tfidf, alist).get_top_results_tfidf(query)
    print(answer[0])


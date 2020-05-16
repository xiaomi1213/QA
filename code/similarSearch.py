# coding: utf-8

"""
基于相似度检索
1 一个文本相似度匹配：在基于检索式系统中核心的部分是计算文本之间的‘相似度‘；
2 词义匹配：提前构建好‘相似的单词‘，搜索阶段使用；
3 倒排表：使用’倒排表’来存储每一个词与出现的文本，以加速搜索速度。
"""


import numpy as np
import queue as Q


class Query():
    def __init__(self,query_vector, X_vectors, alist):
        self.query_vector = query_vector
        self.X_vectors = X_vectors
        self.alist = alist


    def inverted_table(corpus):
        """
        定义一个map结构倒排表，循环所有的单词一遍，然后记录每一个单词所出现的文档。
        :param corpus: 全部文本
        :return: 倒排表
        """
        word_doc = dict()# key:word,value:包含该词的句子序号的列表
        for i, q in enumerate(corpus):
            words = q.strip().split(' ')
            for w in set(words):
                if w not in word_doc:
                    # 没在word_doc中的，建立一个空集合
                    word_doc[w] = set([])
                word_doc[w] = word_doc[w] | set([i])
        return word_doc


    def get_related_words(filename):
        """
        获取相似词
        :param filename:
        :return: related_words
        """
        pass

    # 计算余弦相似度
    def cosineSimilarity(self, vec1, vec2):
        return np.dot(vec1, vec2.T) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))

    # 利用倒排表搜索
    def getCandidate(self, query, inverted_idx, related_words):
        """
        对于句子里的每一个单词，从related_words里提取出跟它意思相近的top 10单词，
        然后根据这些top词从倒排表里提取相关的文档，把所有的文档返回。
        :param self:
        :param query:
        :param inverted_idx:
        :param related_words:
        :return:
        """
        # 根据查询句子中每个词所在的序号列表，求交集
        searched = set()
        for w in query.strip().split(' '):
            if w not in inverted_idx:
                continue
            # 搜索原词所在的序号列表
            if len(searched) == 0:
                searched = set(inverted_idx[w])
            else:
                searched = searched & set(inverted_idx[w])
            # 搜索相似词所在的列表
            if w in related_words:
                for similar in related_words[w]:
                    searched = searched & set(inverted_idx[similar])
        return searched


    def get_top_results_tfidf(self, query):
        """
        给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
        1. 利用倒排表来筛选 candidate （需要使用related_words).
        2. 对于候选文档，计算跟输入问题之间的相似度
        3. 找出相似度最高的top5问题的答案
        """
        top = 5
        query_tfidf = self.query_vector
        X_tfidf = self.X_vectors
        results = Q.PriorityQueue()

        inverted_idx = self.inverted_table()
        related_words = self.get_related_words()

        searched = self.getCandidate(query, inverted_idx, related_words)
        # print(len(searched))
        for candidate in searched:
            # 计算candidate与query的余弦相似度
            result = self.cosineSimilarity(query_tfidf, X_tfidf[candidate])
            # 优先级队列中保存相似度和对应的candidate序号
            # -1保证降序
            results.put((-1 * result, candidate))
        i = 0
        top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
        while i < top and not results.empty():
            top_idxs.append(results.get()[1])
            i += 1
        return np.array(self.alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


    def get_top_results_w2v(self, query):
        """
        给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
        1. 利用倒排表来筛选 candidate （需要使用related_words).
        2. 对于候选文档，计算跟输入问题之间的相似度
        3. 找出相似度最高的top5问题的答案
        """
        # embedding用glove
        top = 5
        query_emb = self.query_vector
        X_w2v = self.X_vectors
        results = Q.PriorityQueue()
        inverted_idx = self.inverted_table()
        related_words = self.get_related_words()
        searched = self.getCandidate(query, inverted_idx, related_words)
        for candidate in searched:
            result = self.cosineSimilarity(query_emb, X_w2v[candidate])
            results.put((-1 * result, candidate))
        top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
        # hint: 利用priority queue来找出top results. 思考为什么可以这么做？
        i = 0
        while i < top and not results.empty():
            top_idxs.append(results.get()[1])
            i += 1
        return np.array(self.alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


    def get_top_results_bert(self, query):
        """
        给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
        1. 利用倒排表来筛选 candidate （需要使用related_words).
        2. 对于候选文档，计算跟输入问题之间的相似度
        3. 找出相似度最高的top5问题的答案
        """
        # embedding用Bert embedding
        top = 5
        # query_emb = np.sum(bert_embedding([query], 'sum')[0][1], axis=0) / len(query.strip().split())
        query_emb = self.query_vector
        X_bert = self.X_vectors
        results = Q.PriorityQueue()
        inverted_idx = self.inverted_table()
        related_words = self.get_related_words()
        searched = self.getCandidate(query, inverted_idx, related_words)
        for candidate in searched:
            result = self.cosineSimilarity(query_emb, X_bert[candidate])
            # print(result)
            results.put((-1 * result, candidate))
        top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
        # hint: 利用priority queue来找出top results. 思考为什么可以这么做？
        i = 0
        while i < top and not results.empty():
            top_idxs.append(results.get()[1])
            i += 1

        return np.array(self.alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案




if __name__ == '__main__':
    pass

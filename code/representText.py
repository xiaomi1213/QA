# coding: utf-8

"""
文本的表示：句子的表示是核心问题，这里会涉及到‘tf-idf’,‘Glove’以及‘BERT Embedding’
"""


import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from bert_embedding import BertEmbedding


class Vectorizer():
    def __ini__(self):
        pass

    # 使用tf-idf表示向量
    def computeTFIDF(self, vocabCount, totalCount, corpus):
        """
        计算每个词的TF-IDF值
        :param vocabCount: 已经统计好的每词的次数
        :param totalCount: 统计好的总次数
        :param corpus: 全部文本
        :return: 全部词的tfidf值
        """

        # TF计算
        tf = np.ones(len(vocabCount))
        for i, (word, fre) in enumerate(vocabCount.items()):
            tf[i] = 1.0 * fre / totalCount

        # IDF计算，没有类别，以句子为一个类
        idf = np.ones(len(vocabCount))
        vocab_list = list(vocabCount.keys())
        for q in corpus:
            words = set(q.strip().split())
            for w in words:
                idf[vocab_list[w]] += 1
        idf /= len(corpus)
        idf = -1.0 * np.log2(idf)
        tf_idf = np.multiply(tf, idf)
        tf_idf = dict(zip(vocab_list,tf_idf))

        return tf_idf

    def vectorizerTFIDF(self, sentence, vocabCount, totalCount, corpus):
        """
        给定句子，计算句子TF-IDF,句子的tfidf是一个1*M的矩阵,M为词表大小，
        不在词表中的词不统计。
        :param sentence: 给定句子
        :param vocabCount: 已经统计好的每词的次数
        :param totalCount: 统计好的总次数
        :param corpus: 全部文本
        :return: 句子的tfidf向量
        """

        tfidf = self.computeTFIDF(vocabCount, totalCount, corpus)

        sentence_tfidf = np.zeros(len(tfidf))
        words = sentence.strip().split(' ')
        for i,w in enumerate(words):
            if w not in tfidf.keys():
                continue
            sentence_tfidf[i] = tfidf[w]
        return sentence_tfidf

    def vectorizerTFIDFs(self, vocabCount, totalCount, corpus):
        """
        对所有句子分别求tfidf
        :param vocabCount:
        :param totalCount:
        :param corpus: 全部文本
        :return:
        """
        tfidf = self.computeTFIDF(vocabCount, totalCount, corpus)
        X_tfidf = np.zeros((len(corpus), len(tfidf)))
        for i, q in enumerate(corpus):
            X_tfidf[i] = self.vectorizerTFIDF(q, vocabCount, totalCount, corpus)
        return X_tfidf


    # 使用wordvec + average pooling
    def loadEmbedding(self, filename):
        # 加载glove模型，转化为word2vec，再加载word2vec模型
        word2vec_temp_file = 'word2vec_temp.txt'
        glove2word2vec(filename, word2vec_temp_file)
        model = KeyedVectors.load_word2vec_format(word2vec_temp_file)
        return model

    def computeGloveSentenceEach(self, sentence, embedding):
        # 查找句子中每个词的embedding,将所有embedding进行加和求均值
        emb = np.zeros(200)
        words = sentence.strip().split(' ')
        for w in words:
            if w not in embedding:
                # 没有lookup的即为unknown
                w = 'unknown'
            # emb += embedding.get_vector(w)
            emb += embedding[w]
        return emb / len(words)

    def computeGloveSentence(self, qlist, embedding):
        # 对每一个句子进行求均值的embedding
        X_w2v = np.zeros((len(qlist), 200))
        for i, q in enumerate(qlist):
            X_w2v[i] = self.computeGloveSentenceEach(q, embedding)
            # print(X_w2v)
        return X_w2v


    # 使用BERT + average pooling
    def embeddingBERT(self, qlist):
        sentence_embedding = np.ones((len(qlist), 768))
        # 加载Bert模型，model，dataset_name,须指定
        bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
        # 查询所有句子的Bert  embedding
        # all_embedding = []
        # for q in qlist:
        #    all_embedding.append(bert_embedding([q],'sum'))
        all_embedding = bert_embedding(qlist, 'sum')
        for i in range(len(all_embedding)):
            # print(all_embedding[i][1])
            sentence_embedding[i] = np.sum(all_embedding[i][1], axis=0) / len(q.strip().split(' '))
            if i == 0:
                print(sentence_embedding[i])

        X_bert = sentence_embedding  # 每一个句子的向量结果存放在X_bert矩阵里。行数为句子的总个数，列数为一个句子embedding大小。
        return X_bert





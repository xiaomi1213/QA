# coding: utf-8

"""
拼写纠错：检查用户输入，如果发现用户拼错了，需要及时在后台改正，然后按照修改后的在库里面搜索；
"""


from nltk.corpus import reuters
import numpy as np
import queue as Q

#1 训练一个语言模型
# 使用nltk自带的reuters数据来训练一个语言模型。 使用add-one smoothing读取语料库的数据
categories = reuters.categories()
corpus = reuters.sents(categories=categories)
#print(corpus[0])
# 循环所有的语料库并构建bigram probability. bigram[word1][word2]: 在word1出现的情况下下一个是word2的概率。
new_corpus = []
for sent in corpus:
    #句子前后加入<s>,</s>表示开始和结束
    new_corpus.append(['<s> '] + sent + [' </s>'])
# print(new_corpus[0])
word2id = dict()
id2word = dict()
for sent in new_corpus:
    for w in sent:
        w = w.lower()
        if w in word2id:
            continue
        id2word[len(word2id)] = w
        word2id[w] = len(word2id)
vocab_size = len(word2id)
count_uni = np.zeros(vocab_size)
count_bi = np.zeros((vocab_size,vocab_size))
#writeVocab(word2id,"lm_vocab.txt")
for sent in new_corpus:
    for i,w in enumerate(sent):
        w = w.lower()
        count_uni[word2id[w]] += 1
        if i < len(sent) - 1:
            count_bi[word2id[w],word2id[sent[i + 1].lower()]] += 1
print("unigram done")
bigram = np.zeros((vocab_size,vocab_size))
#计算bigram LM，有bigram统计值的加一除以|vocab|+uni统计值，没有统计值,
#1 除以 |vocab|+uni统计值
for i in range(vocab_size):
    for j in range(vocab_size):
        if count_bi[i,j] == 0:
            bigram[i,j] = 1.0 / (vocab_size + count_uni[i])
        else:
            bigram[i,j] = (1.0 + count_bi[i,j]) / (vocab_size + count_uni[i])
def checkLM(word1,word2):
    if word1.lower() in word2id and word2.lower() in word2id:
        return bigram[word2id[word1.lower()],word2id[word2.lower()]]
    else:
        return 0.0
print(checkLM('I','like'))


#2 构建Channel Probs
# 构建channel probability
channel = {}
#读取文件，格式为w1:w2,w3..
#w1为正确词，w2,w3...为错误词
#没有给出不同w2-wn的概率，暂时按等概率处理
for line in open('spell-errors.txt'):
    (correct,error) = line.strip().split(':')
    errors = error.split(',')
    errorProb = dict()
    for e in errors:
        errorProb[e.strip()] = 1.0 / len(errors)
    channel[correct.strip()] = errorProb




#3 根据错别字生成所有候选集合, 给定一个错误的单词，首先生成跟这个单词距离为1或者2的所有的候选集合。
def filter(words):
    #将不在词表中的词过滤
    new_words = []
    for w in words:
        if w in word2id:
            new_words.append(w)
    return set(new_words)

def generate_candidates1(word):
    #生成DTW距离为1的词，
    #对于英语来说，插入，替换，删除26个字母
    chars = 'abcdefghijklmnopqrstuvwxyz'
    words = set([])
    #insert 1
    words = set(word[0:i] + chars[j] + word[i:] for i in range(len(word)) for j in range(len(chars)))
    #sub 1
    words = words | set(word[0:i] + chars[j] + word[i+1:] for i in range(len(word)) for j in range(len(chars)))
    #delete 1
    words = words | set(word[0:i] + word[i + 1:] for i in range(len(chars)))
    #交换相邻
    #print(set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1,len(word))))
    words = words | set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1,len(word)))
    #将不在词表中的词去掉
    words = filter(words)
    #去掉word本身
    if word in words:
        words.remove(word)
    return words

def generate_candidates(word):
    # 基于拼写错误的单词，生成跟它的编辑距离为1或者2的单词，并通过词典库的过滤。
    # 只留写法上正确的单词。
    words = generate_candidates1(word)
    words2 = set([])
    for word in words:
        #将距离为1词，再分别计算距离为1的词，
        #作为距离为2的词候选
        words2 = generate_candidates1(word)
    #过滤掉不在词表中的词
    words2 = filter(words)
    #距离为1，2的词合并列表
    words = words  | words2
    return words


#4 给定一个输入，如果有错误需要纠正
def word_corrector(word,context):
    word = word.lower()
    candidate = generate_candidates(word)
    if len(candidate) == 0:
        return word
    correctors = Q.PriorityQueue()
    for w in candidate:
        if w in channel and word in channel[w] and w in word2id and context[0].lower() in word2id and context[1].lower() in word2id:
            probility = np.log(channel[w][word] + 0.0001) +             np.log(bigram[word2id[context[0].lower()],word2id[w]]) +             np.log(bigram[word2id[context[1].lower()],word2id[w]])
            correctors.put((-1 * probility,w))
    if correctors.empty():
        return word
    return correctors.get()[1]



#5 基于拼写纠错算法，实现用户输入自动矫正
def spell_corrector(line):
    # 1. 首先做分词，然后把``line``表示成``tokens``
    # 2. 循环每一token, 然后判断是否存在词库里。如果不存在就意味着是拼写错误的，需要修正。
    #    修正的过程就使用``noisy channel model``, 然后从而找出最好的修正之后的结果。
    new_words = []
    words = ['<s>'] + line.strip().lower().split(' ') + ['</s>']
    for i,word in enumerate(words):
        if i == len(words) - 1:
            break
        word = word.lower()
        if word not in word2id:
            #认为错误，需要修正，句子前后加了<s>,</s>
            #不在词表中词,肯定位于[1,len - 2]之间
            new_words.append(word_corrector(word,(words[i - 1].lower(),words[i + 1].lower())))
        else:
            new_words.append(word)
    newline = ' '.join(new_words[1:])
    return newline   # 修正之后的结果，假如用户输入没有问题，那这时候``newline = line``



if __name__ == "__main__":
    words = generate_candidates('strat')
    print(words)

    word = word_corrector('strat', ('to', 'in'))
    print(word)

    test_query1 = "When did Beyonce strat becoming popular"  # 拼写错误的
    #result:in the late 1990s
    test_query2 = "What counted for more of the poplation change"  # 拼写错误的
    #result:births and deaths
    test_query1 = spell_corrector(test_query1)
    test_query2 = spell_corrector(test_query2)
    print(test_query1)
    print(test_query2)

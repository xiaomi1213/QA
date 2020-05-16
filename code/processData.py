# coding: utf-8

"""
读取及清洗文本：需要从相应的文件里导入(问题，答案)对，涉及到停用词过滤、拼写纠错、数字符号过滤等工作。
"""

import json
from nltk.corpus import stopwords
import codecs
import re


#1 文本的读取
def read_corpus(filename = './data/train-v2.0.json'):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    :return:
    """

    qlist = []
    alist = []
    filename = filename
    datas = json.load(open(filename,'r'))
    data = datas['data']
    for d in data:
        paragraph = d['paragraphs']
        for p in paragraph:
            qas = p['qas']
            for qa in qas:
                #print(qa)
                #处理is_impossible为True时answers空
                if(not qa['is_impossible']):
                    qlist.append(qa['question'])
                    alist.append(qa['answers'][0]['text'])
    #print(qlist[0])
    #print(alist[0])
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist

#2 清洗文本
def tokenizer(ori_list):
    #以标点符号处理分词
    SYMBOLS = re.compile('[\s;\"\",.!?\\/\[\]\{\}\(\)-]+')
    new_list = []
    for q in ori_list:
        words = SYMBOLS.split(q.lower().strip())
        new_list.append(' '.join(words))
    return new_list

def removeStopWord(ori_list):
    new_list = []
    #nltk中stopwords包含what等，但是在QA问题中，这算关键词，所以不看作关键词
    restored = ['what','when','which','how','who','where']
    english_stop_words = list(set(stopwords.words('english')))
    for w in restored:
        english_stop_words.remove(w)
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if w not in english_stop_words])
        new_list.append(sentence)
    return new_list

def removeLowFrequence(ori_list,vocabulary,thres = 10):
    #根据thres筛选词表，小于thres的词去掉
    new_list = []
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if vocabulary[w] >= thres])
        new_list.append(sentence)
    return new_list

def replaceDigits(ori_list,replace = '#number'):
    #将数字统一默认替换为#number
    DIGITS = re.compile('\d+')
    new_list = []
    for q in ori_list:
        q = DIGITS.sub(replace,q)
        new_list.append(q)
    return new_list

def cleanData(ori_list):
    new_list = tokenizer(ori_list)

    new_list = removeStopWord(new_list)

    new_list = replaceDigits(new_list)

    return new_list


#3 输出存储
def createVocab(ori_list):
    count = 0
    vocab_count = dict()
    for q in ori_list:
        words = q.strip().split(' ')
        count += len(words)
        for w in words:
            if w in vocab_count:
                vocab_count[w] += 1
            else:
                vocab_count[w] = 1
    return vocab_count,count


def writeFile(oriList,filename):
    with codecs.open(filename,'w','utf8') as Fout:
        for q in oriList:
            Fout.write(q + u'\n')


def writeVocab(vocabulary,filename):
    sortedList = sorted(vocabulary.items(),key = lambda d:d[1])
    with codecs.open(filename,'w','utf8') as Fout:
        for (w,c) in sortedList:
            Fout.write(w + u':' + str(c) + u'\n')



if __name__ == '__main__':

    qlist,alist = read_corpus()

    new_list = tokenizer(qlist)
    # writeFile(qlist,'ori.txt')

    new_list = removeStopWord(new_list)
    # writeFile(new_list,'removeStop.txt')

    new_list = replaceDigits(new_list)
    # writeFile(new_list,'removeDigts.txt')

    vocabulary, count = createVocab(new_list)
    new_list = removeLowFrequence(new_list, vocabulary, 5)
    writeFile(new_list,'lowFrequence.txt')

    vocabulary2, count2 = createVocab(new_list)

    writeVocab(vocabulary2, "train.vocab")
    print(new_list[0])
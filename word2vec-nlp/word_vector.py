# _*_ coding: utf-8 _*_

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier

def load_data():
    train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')
    test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')
    unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')

    # 影评数量一共100,000条
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size)
    return train,test,unlabeled_train

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    将影评切分成句子，将每句都进行分词
    :param review: 一条影评
    :param tokenizer:标记器
    :param remove_stopwords:是否去除停用词
    :return:句子列表，每个句子都是一个词列表
    '''

    # print review 运行时建议打开
    # 使用NLTK tokenizer将段落切分成句
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

def review_to_wordlist( review, remove_stopwords=False ):
    '''
    将一条影评转换成词列表
    :param review:
    :param remove_stopwords:
    :return:
    '''
    review_text = BeautifulSoup(review,"html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

def train_the_model(train,unlabeled_train):
    sentences = []
    print "将有标签的数据集的影评切分成句子，再切分成词..."
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print "将无标签的数据集的影评切分成句子，再切分成词..."
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    # 确定model已经训练完成不再update时，可对model进行锁定，预载了相似度矩阵，能够提高后面的查询速度，以后model是只读的。
    model.init_sims(replace=True)

    # 保存模型，可用Word2Vec.load()加载
    model_name = "300features_40minwords_10context"
    model.save("data/"+model_name)

def makeFeatureVec(words, model, num_features):
    '''
    将一条评论转换为特征向量
    :param words: 影评
    :param model: word2vec模型
    :param num_features: 词向量大小
    :return:
    '''
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word 是model的词表中所有单词
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    '''
    :param reviews: 影评集合a set of reviews ，each one a list of words
    :param model: word2vec model
    :param num_features: 特征向量大小
    :return: return a 2D numpy array
    '''
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       counter = counter + 1.
    return reviewFeatureVecs

nltk.download()
# 使用nltk.data的sent_tokenize工具对文本进行句子标记（句子切分）
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
train,test,unlabeled_train = load_data()
train_the_model(train,unlabeled_train)
model=word2vec.load("data/300features_40minwords_10context")
# print type(model.wv.syn0)
# print model.wv.syn0.shape
# print model["flower"]
print "为训练数据创建平均特征向量..."
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, model.vector_size )

print "为测试数据创建平均特征向量..."
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, model.vector_size )

forest = RandomForestClassifier( n_estimators = 100 )

print "训练模型..."
forest = forest.fit( trainDataVecs, train["sentiment"] )
result = forest.predict( testDataVecs )
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "data/Word2Vec_AverageVectors.csv", index=False, quoting=3 )
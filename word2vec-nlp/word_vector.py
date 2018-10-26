# _*_ coding: utf-8 _*_

import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def load_data():
    train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')
    test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')
    unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')

    # 影评数量一共100,000条
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size)
    return train,test,unlabeled_train

# Download the punkt tokenizer for sentence splitting
import nltk.data
# nltk.download()
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    将影评切分成句子，将每句都进行分词
    :param review: 一条影评
    :param tokenizer:标记器
    :param remove_stopwords:是否去除停用词
    :return:句子列表，每个句子都是一个词列表
    '''

    # print review 运行时建议打开
    # Use the NLTK tokenizer to split the paragraph into sentences
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

    # Initialize and train the model (this will take some time)
    from gensim.models import word2vec
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # 保存模型，可用Word2Vec.load()加载
    model_name = "300features_40minwords_10context1"
    model.save(model_name)

train,test,unlabeled_train = load_data()
train_the_model(train,unlabeled_train)
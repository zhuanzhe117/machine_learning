# _*_ coding: utf-8 _*_

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def review_to_words( raw_review ):

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4.在Python中查找set比list快
    stops = set(stopwords.words("english"))

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    return( " ".join( meaningful_words ))

def preprocess_data(train):
    num_reviews = train["review"].size
    print "开始清理和分析训练集的影评...\n"
    clean_train_reviews = []

    for i in range(0, num_reviews):
        if ((i + 1) % 1000 == 0):
            print "Review %d of %d\n" % (i + 1, num_reviews)
        clean_train_reviews.append(review_to_words(train["review"][i]))

    print "Creating the bag of words...\n"
    # 初始化CountVectorizer对象, 是scikit-learn的词袋模型工具
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=5000)

    # fit_transform() : First, it fits the model and learns the vocabulary;
    #  second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    train_data_features = train_data_features.toarray()
    return train_data_features,vectorizer

def train_model_classify(train_data_features,vectorizer):
    print "Training the random forest..."

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])

    test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)
    print test.shape
    num_reviews = len(test["review"])
    clean_test_reviews = []

    print "清理和分析测试集的影评...\n"
    for i in xrange(0, num_reviews):
        if ((i + 1) % 1000 == 0):
            print "Review %d of %d\n" % (i + 1, num_reviews)
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    result = forest.predict(test_data_features)
    df = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # Use pandas to write the comma-separated output file
    df.to_csv("data/Bag_of_Words_Result.csv", index=False, quoting=3)

if __name__=="__main__":
    train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    print train.shape
    print train.columns.values
    train_data_features, vectorizer = preprocess_data(train)
    train_model_classify(train_data_features, vectorizer)


# _*_ coding: utf-8 _*_

from word_vector import *
from gensim.models import Word2Vec
model=Word2Vec.load("300features_40minwords_10context")
# print type(model.wv.syn0)
# print model.wv.syn0.shape
# print model["flower"]

import numpy as np
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       counter = counter + 1.
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word removal.
print "Creating average feature vecs for train reviews"
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, model.vector_size )

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, model.vector_size )

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )
result = forest.predict( testDataVecs )
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
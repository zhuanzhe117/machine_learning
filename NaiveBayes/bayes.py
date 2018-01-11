#encoding=utf-8

from numpy import *

#词表到向量的转换函数
def loadDataSet():
    postingList=[
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthlsess','dog','food','stupid']
    ]
    classVec=[0,1,0,1,0,1] #0代表正常言论，1代表侮辱性文字
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary! " % word
    return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords);p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # print p1Num
    # print p1Denom
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive


if __name__ == '__main__':
    listOPosts,listClasses = loadDataSet()
    print listOPosts
    print listClasses
    myVocabList = createVocabList(listOPosts)
    print myVocabList
    # tempVec = setOfWords2Vec(vocabList,listOPosts[0])
    # print tempVec
    trainMat =[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print trainMat

    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    print p0V
    print p1V
    print pAb
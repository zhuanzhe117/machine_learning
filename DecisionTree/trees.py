#encoding=utf-8
from math import log

#计算给定数据集的香农熵
def calcShannonEnt (dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print labelCounts
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print splitDataSet(myDat,0,1)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X, y = iris.data, iris.target
    y[y != 1] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    from sklearn.dummy import DummyClassifier
    from sklearn.svm import SVC

    clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(X_train, y_train)
    DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
    clf.score(X_test, y_test)

    clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

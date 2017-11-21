#encoding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range (k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#从文本文件（datingTestSet2.txt）中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 移除字符串头尾指定的字符，默认为空格
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat [index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat ,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.90
    datingDataMat ,datingLabels = file2matrix('data\datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #哪些数据用于测试，那些用于分类器的训练样本
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print"the classifier came back with : %d, the real answer is : %d" % (classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]) : errorCount += 1.0
    print "the total error rate is : %f" % (errorCount /float(numTestVecs))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream= float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('data/datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,norMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]

if __name__ == '__main__':
    #样本训练集
    # group,labels = createDataSet()
    #对新数据[0,0]进行分类，结果为B标签
    # print(classify0([0,0],group,labels,3))
    # #解析文本文件
    # datingDataMat ,datingLabels = file2matrix("data\datingTestSet2.txt")
    # #归一化数值
    # normMat,ranges,minVals = autoNorm(datingDataMat)
    # print normMat,ranges,minVals
    # #创建散点图
    # fig = plt.figure()
    # #将画布分割成1行1列，图像画在从左到右从上到下的第1块
    # ax = fig.add_subplot(111)
    # #[:,1]表示对二维数组，取第一维的所有数据，第二维中的第1个数据（从0开始），也就是所有行的第1个数据
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.xlabel("video and games")#玩视频游戏所耗时间百分比
    # plt.ylabel("ice cream")#每周消费的冰淇淋公升数
    # plt.show()
    datingClassTest()
    # classifyPerson()
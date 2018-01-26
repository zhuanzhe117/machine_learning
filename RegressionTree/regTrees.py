#encoding=utf8
from numpy import *

#CART算法代码实现
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

#生成叶节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
#在给定数据上计算目标变量的平方误差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def createTree(dataSet, leafType = regLeaf, errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val #满足停止条件时返回叶节点值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


#回归树的切分函数
def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) ==1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS :
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS ) < tolS :
        return None,leafType(dataSet)
    mat0 ,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

if __name__ == '__main__':
    # testMat = mat(eye(4))
    # print testMat
    # mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    # print mat0

    myDat = loadDataSet('data/ex00.txt')
    myMat = mat(myDat)
    createTree(myMat)
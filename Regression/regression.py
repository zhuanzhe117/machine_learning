#encoding=utf-8
from numpy import *
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))  -1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
#标准回归函数
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (xMat.T * yMat)
    #ws = linalg.solve(xTx,xMat.T*yMat)  使用函数求解未知矩阵
    return ws

if __name__ == '__main__':
    xArr ,yArr = loadDataSet('data/ex0.txt')
    weights = standRegres(xArr,yArr)
    import matplotlib.pyplot as plt
    xMat = mat(xArr)
    yMat = mat(yArr) #真实值

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    xCopy = xMat.copy() #
    xCopy.sort(0)
    yHat = xCopy * weights #预测值
    ax.plot(xCopy[:,1],yHat)

    plt.show()
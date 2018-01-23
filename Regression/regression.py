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

#局部加权线性回归函数
#先求公式中的W（待测点附近的每个点的权重），即weights，然后求回归系数ws，返回待测点的y值
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:] #权重值大小以指数级衰减
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0 * k ** 2)) #运用高斯核对附近点赋予权重，点越近权重越大
    xTx = xMat.T * (weights*xMat)
    if linalg.det(xTx) == 0.0: #奇异矩阵
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#测试局部加权函数，传入多个待测点
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__ == '__main__':
    xArr ,yArr = loadDataSet('data/ex0.txt')
    # weights = standRegres(xArr,yArr)
    # import matplotlib.pyplot as plt
    # #画样本点
    # xMat = mat(xArr)
    # yMat = mat(yArr) #真实值
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    #
    # # 画预测点，将直线上的数据点按照升序排列
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * weights #预测值
    # ax.plot(xCopy[:,1],yHat)
    # plt.show()

    #局部加权效果
    yHat = lwlrTest(xArr,xArr,yArr,0.01)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()


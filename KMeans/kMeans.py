# _*_ coding: utf-8 _*_

import sys
import os
from numpy import *

reload(sys)
sys.setdefaultencoding('utf-8')

#文件数据转换为矩阵
def file2matrix(path,delimiter):
    fp = open(path,'rb')
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist = [map(eval, row.split(delimiter)) for row in rowlist if row.strip()]
    return mat(recordlist)

#欧式距离公式
def disEclud(vecA,vecB):
    return linalg.norm(vecA-vecB)

#随机生成聚类中心
def randCenters(dataSet ,k):
    n = shape(dataSet)[1]
    clustercents = mat(zeros((k,n)))
    for col in xrange(n):
        mincol = min(dataSet[:,col])
        maxcol = max(dataSet[:,col])
        clustercents[:, col] = mat(mincol + float(maxcol - mincol) * random.rand(k, 1))
    return clustercents

import matplotlib.pyplot as plt

# from Recommand_Lib imort *
def color_cluster(dataindx, dataSet, plt, k=4):
    index = 0
    datalen = len(dataindx)
    for indx in range(datalen):
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='blue', marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='green', marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='red', marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='cyan', marker='o')
        index += 1


def drawScatter(plt, mydata, size=20, color='blue', mrkr='o'):
    plt.scatter(mydata.T[0].tolist(), mydata.T[1].tolist(), s=size, c=color, marker=mrkr)

def kMeans(dataSet, k, distMeas=disEclud, createCent=randCenters):
    m = shape(dataSet)[0]
    ClustDist = mat(zeros((m, 2)))
    clustercents = randCenters(dataSet, k)
    flag = True
    while flag:
        flag = False
        for i in range(m):
            # 遍历k个聚类中心，获取最短距离
            distlist = [disEclud(clustercents[j, :], dataSet[i, :]) for j in range(k)]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)
            if ClustDist[i, 0] != minIndex:
                flag = True
            ClustDist[i, :] = minIndex, minDist**2

        for cent in range(k):
            ptsInClust = dataSet[nonzero(ClustDist[:, 0].A == cent)[0]]
            clustercents[cent, :] = mean(ptsInClust, axis=0)
            print "clustercents:\n", clustercents
    return clustercents,ClustDist

if __name__=="__main__":
    dataMat = file2matrix("data/4k2_far.txt", "\t")  # 构建数据集
    dataSet = mat(dataMat[:, 1:])
    clustercents,ClustDist = kMeans(dataSet,4)
    color_cluster(ClustDist[:, 0:1], dataSet, plt)
    drawScatter(plt, clustercents, size=60, color='red', mrkr='D')
    plt.show()
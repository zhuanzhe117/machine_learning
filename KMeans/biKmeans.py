# _*_ coding: utf-8 _*_

#为了克服K-均值算法收敛于局部最小值的问题，有人提出二分K-均值算法，它的聚类效果很好
from kMeans import *

def biKmeans(dataSet, k, distMeas=disEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): #直到满足质心数为k，否则一直划分
        lowestSSE = inf
        for i in range(len(centList)): #这个循环会得到最应该划分的簇，并把这个簇一分为二
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit +sseNnotSplit = ",sseSplit+sseNotSplit
            print "lowestSSE: ",lowestSSE
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
# if __name__=="__main__":
#     dataMat = file2matrix("data/4k2_far.txt", "\t")  # 构建数据集
#     dataSet = mat(dataMat[:, 1:])
#     clustercents, ClustDist = biKmeans(dataSet,4)
#     color_cluster(ClustDist[:, 0:1], dataSet, plt)
#     drawScatter(plt, clustercents, size=60, color='red', mrkr='D')
#     plt.show()
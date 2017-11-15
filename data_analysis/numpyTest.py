#encoding=utf-8
import numpy as np
from numpy.linalg import *
def main():
    # lst=[[1,3,5],[2,4,6]]
    # print(type(lst))
    #
    # np_lst = np.array(lst)
    # print(type(np_lst))
    #列表，数组，数据类型
    # np_lst = np.array(lst,dtype = np.float)
    # print(np_lst.shape)
    # print(np_lst.ndim)
    # print(np_lst.dtype)
    # print(np_lst.itemsize)
    # print(np_lst.size)
    #初始化矩阵
     # print(np.zeros([2,3]))
    # print(np.ones([3,5]))
    #随机数
    # print("Rand:")
    # print(np.random.rand(2,4))
    # print(np.random.randint(1,20,3))
    # print("choice:")
    # print(np.random.choice([10,20,30]))
    # print(np.random.beta(1,10,50))
    #数学函数
    # lst = np.arange(1 ,11).reshape([2,-1])
    # print(lst)
    # print(np.exp(lst))
    # print(np.exp2(lst))
    # print(np.sqrt(lst))
    # print(np.sin(lst))
    # print("log")
    # print(np.log(lst))

    #矩阵，行列式
    print(np.eye(3))
    lst = np.array([[1,2],
                    [3,4]])
    print(lst)
    # print(inv(lst))
    # print(lst.transpose())
    # print(det(lst))
    y = np.array([[5],[7]])
    print(solve(lst,y))

if __name__ == '__main__':
    main()

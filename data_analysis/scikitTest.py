#encoding=utf-8
import numpy as np
import pandas as pd

def main():
    #Pre-processing
    #数据样本
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(iris)
    # print(len(iris["data"]))

    #对数据进行 预处理
    from sklearn.cross_validation import train_test_split
    train_data,test_data,train_target,test_target = train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)
    # print(train_data)
    #Model
    from sklearn import tree
    #决策树
    clf=tree.DecisionTreeClassifier(criterion="entropy")
    #训练集进行训练
    clf.fit(train_data,train_target)
    #预测
    y_pred = clf.predict(test_data)

    #Verify
    from sklearn import metrics
    print(metrics.accuracy_score(y_true=test_target,y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred))

    with open("./data/tree.dot","w") as fw:
        tree.export_graphviz(clf,out_file=fw)
if __name__ == '__main__':
        main()
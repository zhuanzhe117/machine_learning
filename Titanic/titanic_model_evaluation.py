#encoding=utf-8
'''
kaggle：泰坦尼克号之灾，预测谁能生还： 模型选择与评估
'''

# ******************************画出模型的学习曲线*******************************
import pandas as pd
import matplotlib
matplotlib.use('qt5agg')
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn import linear_model

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve
    estimator : 你用的分类器
    title : 表格的标题
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    # return midpoint, diff
    print midpoint
    print diff

#********************************交叉验证********************************

# from sklearn import linear_model
# from sklearn.model_selection import *
# # 简单看看打分情况
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.as_matrix()[:,1:]
# y = all_data.as_matrix()[:,0]
# print cross_val_score(clf, X, y, cv=5)
#
# # 分割数据
# split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
#
# # 对cross validation数据进行预测
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:,1:])
# # split_cv[ predictions != cv_df.as_matrix()[:,0] ].drop()
#
# # 去除预测错误的case看原始dataframe数据
# #split_cv['PredictResult'] = predictions
# origin_data_train = pd.read_csv("data/Train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
# print bad_cases

if __name__=="__main__":
    # df = pd.read_csv("data/train_handled.csv")
    # train_data = df.as_matrix()
    # y = train_data[:,1]
    # x = train_data[:,2:]
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # plot_learning_curve(clf, u"学习曲线", x, y)

    import sklearn
    print sklearn.__version__
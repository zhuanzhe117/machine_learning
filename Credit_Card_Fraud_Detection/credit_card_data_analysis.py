#encoding=utf-8
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# data_cr = pd.read_csv("D:/materials/dataset/Credit_Card_Fraud_Detection/data/creditcard.csv")
# # 目标变量分布可视化
# fig, axs = plt.subplots(1,2,figsize=(14,7))
# sns.countplot(x='Class',data=data_cr,ax=axs[0])
# axs[0].set_title("Frequency of each Class")
# data_cr['Class'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
# axs[1].set_title("Percentage of each Class")
# plt.show()
#
# # 查看目标列的情况
# print (data_cr.groupby('Class').size())
# # 特征衍生 单位转换
# data_cr['Hour'] =data_cr["Time"].apply(lambda x : divmod(x, 3600)[0])
#
# #特征选择（数据探索）查看信用卡正常用户与被盗刷用户之间的区别
# Xfraud = data_cr.loc[data_cr["Class"] == 1]
# XnonFraud = data_cr.loc[data_cr["Class"] == 0]
#
# correlationNonFraud = XnonFraud.loc[:, data_cr.columns != 'Class'].corr()
# mask = np.zeros_like(correlationNonFraud)
# indices = np.triu_indices_from(correlationNonFraud)
# mask[indices] = True
#
# grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
# f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14, 9))
#
# cmap = sns.diverging_palette(220, 8, as_cmap=True)
# ax1 = sns.heatmap(correlationNonFraud, ax=ax1, vmin=-1, vmax=1,
#                   cmap=cmap, square=False, linewidths=0.5, mask=mask, cbar=False)
# ax1.set_xticklabels(ax1.get_xticklabels(), size=9)
# ax1.set_yticklabels(ax1.get_yticklabels(), size=9)
# ax1.set_title('Normal', size=20)
#
# correlationFraud = Xfraud.loc[:, data_cr.columns != 'Class'].corr()
# ax2 = sns.heatmap(correlationFraud, vmin=-1, vmax=1, cmap=cmap,
#                   ax=ax2, square=False, linewidths=0.5, mask=mask, yticklabels=False,
#                   cbar_ax=cbar_ax, cbar_kws={'orientation': 'vertical','ticks': [-1, -0.5, 0, 0.5, 1]})
# ax2.set_xticklabels(ax2.get_xticklabels(), size=9)
# ax2.set_title('Fraud', size=20)
#
# cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=14)
# #由以上热力图知：欺诈类样本的特征V1、V2、V3、V4、V5、V6、V7、V9、V10、V11、V12、V14、V16、V17、V18以及V19之间成强相关关系
#
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,4))
# bins = 30
#
# ax1.hist(data_cr["Amount"][data_cr["Class"]== 1], bins = bins)
# ax1.set_title('Fraud')
#
# ax2.hist(data_cr["Amount"][data_cr["Class"] == 0], bins = bins)
# ax2.set_title('Normal')
#
# plt.xlabel('Amount ($)')
# plt.ylabel('Number of Transactions')
# plt.yscale('log')
# plt.show()
# #由以上直方图知：欺诈类更偏向选择小额消费
# sns.factorplot(x="Hour", data=data_cr, kind="count",  palette="ocean", size=6, aspect=3)
# #由上图知：每天早上9点到晚上11点之间是信用卡消费的高频时间段
#
# f, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
# ax1.scatter(data_cr["Hour"][data_cr["Class"] == 1], data_cr["Amount"][data_cr["Class"]  == 1],linewidths=0.001)
# ax1.set_title('Fraud')
# ax2.scatter(data_cr["Hour"][data_cr["Class"] == 0], data_cr["Amount"][data_cr["Class"] == 0])
# ax2.set_title('Normal')
#
# plt.xlabel('Time (in Hours)')
# plt.ylabel('Amount')
# plt.show()
# #由上图知：信用卡被盗刷数量案发最高峰在第一天上午11点达到43次，
# # 其余发生信用卡被盗刷案发时间在晚上时间11点至第二早上9点之间，
# # 说明为了不引起卡主注意，更喜欢选择睡觉时间和消费频率较高的时间点作案；
# # 同时，信用卡发生被盗刷的最大值只有2,125.87美元。
#
# print ("Fraud Stats Summary")
# print (data_cr["Amount"][data_cr["Class"] == 1].describe())
# print ()
# print ("Normal Stats Summary")
# print (data_cr["Amount"][data_cr["Class"]  == 0].describe())
#
# #删除无关特征，以及Time，保留离散程度更小的Hour。
# droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
# data_new = data_cr.drop(droplist, axis = 1)
# data_new.to_csv('D:/materials/dataset/Credit_Card_Fraud_Detection/data/creditcard_handled.csv', index=False)



data_new = pd.read_csv("D:/materials/dataset/Credit_Card_Fraud_Detection/data/creditcard_handled.csv")
#利用随机森林的feature importance对特征的重要性进行排序
x_feature = list(data_new.columns)
x_feature.remove('Class')
x_val = data_new[x_feature]
y_val = data_new['Class']

names = data_new[x_feature].columns
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10,random_state=123)
clf.fit(x_val, y_val) #对自变量和因变量进行拟合
# names, clf.feature_importances_
for feature in zip(names, clf.feature_importances_):
    print(feature)

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

## feature importances 可视化##
importances = clf.feature_importances_
feat_names = names
indices = np.argsort(importances)[::-1]
fig = plt.figure(figsize=(10,5))
plt.title("Feature importances by RandomTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
# plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation=90,fontsize=14)
plt.xlim([-1, len(indices)])
plt.tight_layout()
plt.show()
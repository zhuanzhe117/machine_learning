#encoding=utf-8
'''
kaggle案例：泰坦尼克号之灾，预测谁能生还: 数据分析
'''
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

data_train = pd.read_csv("data/Train.csv")

# print data_train.columns
#data_train[data_train.Cabin.notnull()]['Survived'].value_counts()

# print data_train.info()
# print data_train.describe()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qt5agg')
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
# plt.title(u"获救情况 (1为获救)") # puts a title on our graph
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # sets the y axis lable
# plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
# plt.title(u"按年龄看获救分布 (1为获救)")
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'1L', u'2L',u'3L'),loc='best') # sets our legend for our graph.
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.tight_layout(pad=1, w_pad=1, h_pad=1)
# plt.show()


#看看各乘客等级的获救情况
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=False)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
# plt.show()

#看看各登录港口的获救情况
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({ u'未获救':Survived_0,u'获救':Survived_1})
# df.plot(kind='bar', stacked=False)
# plt.title(u"各登录港口乘客的获救情况")
# plt.xlabel(u"登录港口")
# plt.ylabel(u"人数")
# plt.show()

#看看各性别的获救情况
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
# plt.show()
#
# #然后我们再来看看各种舱级别情况下各性别的获救情况
# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels([u"sur", u"not"], rotation=0)
# ax1.legend([u"female/High"], loc='best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"not", u"sur"], rotation=0)
# plt.legend([u"female/Low"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"not", u"sur"], rotation=0)
# plt.legend([u"male/High"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"not", u"sur"], rotation=0)
# plt.legend([u"male/Low"], loc='best')
#
# plt.show()

#堂兄弟、父母、大家族有什么优势
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df

# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df

#cabin船舱分布，缺失值严重
#先在有无Cabin信息这个粗粒度上看看Survived的情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"按Cabin有无看获救情况")
# plt.xlabel(u"Cabin有无")
# plt.ylabel(u"人数")
# plt.show()

#encoding=utf-8

'''
kaggle：房价预测
'''

import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #小数点后三位
warnings.filterwarnings('ignore')
# from subprocess import check_output
# print(check_output(["ls", "data"]).decode("utf8")) #check the files available in the directory

def loadData():
    """
    加载数据集
    :return:
    """
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')
    return data_train,data_test

def deleteOutliers(data_train):
    '''
    观察数据并删除训练集中的异常点
    :param data_train:
    :return:
    '''
    fig, ax = plt.subplots()
    ax.scatter(x=data_train['GrLivArea'], y=data_train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()

    train = data_train.drop(data_train[(data_train['GrLivArea'] > 4000) & (data_train['SalePrice'] < 300000)].index)
    return train

def temp(trainY):
    '''
    检查数据是否符合正态分布
    :param trainY: 房价
    :return:
    '''
    sns.distplot(trainY, fit=norm)#直方图
    (mu, sigma) = norm.fit(trainY)# 拟合参数
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    # 画出QQ-plot，用以检测样本数据是否近似于正态分布
    fig = plt.figure()
    res = stats.probplot(trainY, plot=plt)
    plt.show()

def check_missing_data(all_data):
    '''
    缺失数据可视化
    :param all_data:
    :return:
    '''
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})

    # 缺失数据百分比降序排列的条形图
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

def main():
    data_train,data_test = loadData()

    train = deleteOutliers(data_train)

    trainX = train.drop(['Id', 'SalePrice'], axis=1)
    trainY =train["SalePrice"]
    test_ID = data_test['Id']
    testX = data_test.drop(['Id'],axis=1)

    temp(trainY)
    #调用numpy的log1p函数对房价进行处理
    trainY = np.log1p(trainY)
    temp(trainY)

    dataSet = pd.concat([trainX,testX],axis=0)
    check_missing_data(dataSet)

    # 绘制相关性图，各特征与房价的关系
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)



def imputingMissingData(all_data):
    '''
    填补缺失值
    :param all_data:
    :return:
    '''
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    #按neighborhood分组，用所有相同邻居的LotFrontage中位数填充缺失值
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna("None")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna("None")

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    # 缺失值很少，也不是因为不存在而为Nan，可用众数补全
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    # Utilities全部取值都是 "AllPub"，可以直接删掉
    all_data = all_data.drop(['Utilities'], axis=1)

    # Functional : data description says NA means typical
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    # KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

    # SaleType : Fill in again with most frequent which is "WD"
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    # MSSubClass : Na most likely means No building class. We can replace missing values with None
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    # Is there any remaining missing value ?

    #Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head()

    # Missing Ratio

    # 1、有许多特征实际上是类别型的特征，但给出来的是数字。比如MSSubClass，是评价房子种类的一个特征，给出的是10-100的数字，但实际上是类别，所以我们需要将其转化为字符串类别。

    #MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # 2、接下来 LabelEncoder，对文本类别的特征进行编号。

    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape
    print('Shape all_data: {}'.format(all_data.shape))

    # 3、接下来添加一个重要的特征，因为我们实际在购买房子的时候会考虑总面积的大小，但是此数据集中并没有包含此数据。总面积等于地下室面积+1层面积+2层面积。

    # Adding total sqfootage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # 4、我们对房价进行分析，不符合正态分布我们将其log转换，使其符合正态分布。那么偏离正态分布太多的特征我们也对它进行转化：

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)


    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    #5、将类别特征进行哑变量转化：
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)

main()
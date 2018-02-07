#encoding=utf-8
import pandas as pd #数据分析
import numpy as np #科学计算

data_train = pd.read_csv("Train.csv")
data_train['Sex_Pclass'] = data_train.Sex + "_" + data_train.Pclass.map(str)

from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_train['Sex_Pclass'], prefix='Sex_Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)


#*********************************************************

data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
data_test['Sex_Pclass'] = data_test.Sex + "_" + data_test.Pclass.map(str)
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_test['Sex_Pclass'], prefix= 'Sex_Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test

#*************************************************
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions2.csv", index=False)


#**************************用scikit-learn的Bagging 模型融合*************************
from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions2.csv", index=False)

#****************************用别的分类器解决这个问题**********************************
import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib

##Read configuration parameters

train_file = "train.csv"
MODEL_PATH = "./"
test_file = "test.csv"
SUBMISSION_PATH = "./"
seed = 0

print train_file, seed


# 输出得分
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# 清理和处理数据
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def clean_and_munge_data(df):
    # 处理缺省值
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    # 处理一下名字，生成Title字段
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    # 处理特殊的称呼，全处理成mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)

    # 看看家族是否够大，咳咳
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Family'] = df['SibSp'] * df['Parch']

    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')

    df.loc[df.Cabin.isnull() == True, 'Cabin'] = 0.5
    df.loc[df.Cabin.isnull() == False, 'Cabin'] = 1.5

    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

    # Age times class

    df['AgeClass'] = df['AgeFill'] * df['Pclass']
    df['ClassFare'] = df['Pclass'] * df['Fare_Per_Person']

    df['HighLow'] = df['Pclass']
    df.loc[(df.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df.loc[(df.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

    le.fit(df['Ticket'])
    x_Ticket = le.transform(df['Ticket'])
    df['Ticket'] = x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title = le.transform(df['Title'])
    df['Title'] = x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl = le.transform(df['HighLow'])
    df['HighLow'] = x_hl.astype(np.float)

    le.fit(df['AgeCat'])
    x_age = le.transform(df['AgeCat'])
    df['AgeCat'] = x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb = le.transform(df['Embarked'])
    df['Embarked'] = x_emb.astype(np.float)

    df = df.drop(['PassengerId', 'Name', 'Age', 'Cabin'], axis=1)  # remove Name,Age and PassengerId

    return df

# 读取数据
traindf = pd.read_csv(train_file)
##清洗数据
df = clean_and_munge_data(traindf)
########################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print y_train.shape, x_train.shape

##选择训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# 初始化分类器
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

###grid search找到最好的参数
param_grid = dict()
##创建分类pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy', \
                           cv=StratifiedShuffleSplit(Y_train, n_iter=10, test_size=0.2, train_size=None, indices=None, \
                                                     random_state=seed, n_iterations=None)).fit(X_train, Y_train)
# 对结果打分
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)

print('-----grid search end------------')
print ('on all train set')
scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
print scores.mean(), scores
print ('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test, cv=3, scoring='accuracy')
print scores.mean(), scores

# 对结果打分

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train)))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test)))

model_file = MODEL_PATH + 'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)
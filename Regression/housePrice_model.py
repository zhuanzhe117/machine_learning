#encoding=utf-8
'''
kaggle：房价预测-模型选择与融合
'''
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

n_folds = 5
def load_train_dataSet():
    train = pd.read_csv("data/train_handled.csv")
    trainX = train.drop(['SalePrice'],axis=1)
    trainY = train['SalePrice']
    return trainX,trainY

def load_test_dataSet():
    test = pd.read_csv("data/test_handled.csv")
    test_ID = test['Id']
    testX = test.drop(['Id'],axis=1)
    return test_ID,testX

def rmsle_cv(model,trainX,trainY):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(trainX.values)
    rmse= np.sqrt(-cross_val_score(model, trainX.values, trainY, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def main():
    trainX,trainY = load_train_dataSet()

    lasso = Lasso(alpha=0.0005, random_state=1)

    ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    GBDT = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    # LightGBM :
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    #**********************对每个模型进行交叉验证得平均分************************
    score = rmsle_cv(lasso,trainX,trainY)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # Lasso score: 0.1115 (0.0074)

    score = rmsle_cv(ENet,trainX,trainY)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # ElasticNet score: 0.1116 (0.0074)

    score = rmsle_cv(KRR,trainX,trainY)
    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # Kernel Ridge score: 0.1153 (0.0075)

    score = rmsle_cv(GBDT,trainX,trainY)
    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # Gradient Boosting score: 0.1167 (0.0084)

    score = rmsle_cv(model_lgb,trainX,trainY)
    print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # LGBM score: 0.1162 (0.0057)

    # ***********************简单地用stacking对多个模型进行融合***********************
    #对四个模型的预测结果求平均值
    averaged_models = AveragingModels(models=(ENet, GBDT, KRR, lasso))
    score = rmsle_cv(averaged_models,trainX,trainY)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # Averaged base models score: 0.1091 (0.0075)

    # ***********************stacking模型融合***********************

    test_ID,testX = load_test_dataSet()

    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBDT, KRR), meta_model=lasso)
    stacked_averaged_models.fit(trainX.values, trainY)
    stacked_train_pred = stacked_averaged_models.predict(trainX.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(testX))
    print("stacked score: ",rmsle(trainY, stacked_train_pred))
    # 0.0781571937916

    model_lgb.fit(trainX, trainY)
    lgb_train_pred = model_lgb.predict(trainX)
    lgb_pred = np.expm1(model_lgb.predict(testX.values))
    print("model_lgb score: ",rmsle(trainY, lgb_train_pred))
    # # 0.0719406222196

    print('RMSE score on train data:')
    print(rmsle(trainY, stacked_train_pred * 0.70 + lgb_train_pred * 0.30))
    # RMSLE score on train data: 0.0752452023077

    ensemble = stacked_pred * 0.70 + lgb_pred * 0.30
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('data/submission.csv', index=False)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    简单地对多个模型进行融合，对每个模型的预测结果取平均值，从而达到更好的预测结果
    '''
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        '''
        对模型集合中的对象进行深拷贝，并训练
        '''
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    #用组件模型预测，对结果求平均值
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 创建一个数组，用于存储每个模型对训练集进行交叉预测的结果
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):#依次取出每一个模型
            for train_index, holdout_index in kfold.split(X, y):#对训练集进行切分
                instance = clone(model)#因为每个模型都会进行5次训练，而且每次训练集都不一样，所以要对未fit的模型克隆一份
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])#训练一个初级分类器
                y_pred = instance.predict(X[holdout_index])#用这个初级分类器预测验证集的结果
                out_of_fold_predictions[holdout_index, i] = y_pred#将这个初级分类器预测所有验证集的结果拼到数组中，最后会成为一列

        # 再使用多个初级分类器对验证集的预测结果作为新的输入，训练元分类器
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #先用self.base_models_中保存的所有的初级分类器（每个模型有5个）对测试数据进行预测，取预测结果的平均值，
    # 作为元分类器的特征，得到最后预测结果
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)#1代表压缩列
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

main()

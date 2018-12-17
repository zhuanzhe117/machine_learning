# -*- coding: utf-8 -*-
# 数据读取与计算
import warnings

import numpy as np
import pandas as pd
# 随机森林与SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
# 数据预处理与模型选择
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
show_bdry = False
show_best_c = False

# 对所有特征做标准化处理，或者对Amount做
def normalize_feature(data, amount_only = False):
    if amount_only:
        data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    else:
        for feature in data.columns[:-1]:
            data[feature] = StandardScaler().fit_transform(data[feature].values.reshape(-1, 1))
    return data

# 数据被切分成训练集和测试集
def split_train_test(fraud_indices, normal_indices, test_size = 0.3):
    number_records_fraud = len(fraud_indices)
    number_records_normal = len(normal_indices)
    test_fraud_end = int(number_records_fraud * test_size)
    test_normal_end = int(number_records_normal * test_size)

    test_fraud_indices = fraud_indices[0:test_fraud_end]
    train_fraud_indices = fraud_indices[test_fraud_end:]

    test_normal_indices = normal_indices[0:test_normal_end]
    train_normal_indices = normal_indices[test_normal_end:]

    return train_normal_indices, train_fraud_indices, test_normal_indices, test_fraud_indices

def getTrainingSample(train_fraud_indices, train_normal_indices, data, train_normal_pos, ratio):
    """
    欠采样
    :param train_fraud_indices: 正类样本数据索引
    :param train_normal_indices: 负类样本数据索引
    :param data: 全部数据
    :param train_normal_pos: 从指定位置开始采样
    :param ratio: 采样比例
    :return:
    """
    train_number_records_fraud = int(ratio*len(train_fraud_indices))
    train_number_records_normal = len(train_normal_indices)
    if train_normal_pos + train_number_records_fraud <= train_number_records_normal:
        small_train_normal_indices = train_normal_indices[train_normal_pos: train_normal_pos + train_number_records_fraud]  # 获取和欺诈类个数一样多的正常类样本
        train_normal_pos = train_normal_pos + train_number_records_fraud
    else:
        # 从指定位置往后取到最后一条数据，然后再从0开始取还差的数量
        small_train_normal_indices = np.concatenate([train_normal_indices[train_normal_pos: train_number_records_normal],
                                                     train_normal_indices[0: train_normal_pos + train_number_records_fraud - train_number_records_normal]])
        train_normal_pos = train_normal_pos + train_number_records_fraud - train_number_records_normal

    under_train_sample_indices = np.concatenate([train_fraud_indices, small_train_normal_indices])
    np.random.shuffle(under_train_sample_indices)

    under_train_sample_data = data.iloc[under_train_sample_indices, :]

    X_train_undersample = under_train_sample_data.ix[:, under_train_sample_data.columns != 'Class']
    y_train_undersample = under_train_sample_data.ix[:, under_train_sample_data.columns == 'Class']

    return X_train_undersample, y_train_undersample, train_normal_pos

# 最简单的KNN模型
def knn_module(X, y, indices, c_param, bdry = None):
    knn = KNeighborsClassifier(n_neighbors=c_param)
    knn.fit(X.iloc[indices[0], :], y.iloc[indices[0], :].values.ravel())
    y_pred_undersample = knn.predict(X.iloc[indices[1], :].values)
    return y_pred_undersample

# 加RBF核的svm
def svm_rbf_module(X, y, indices, c_param, bdry = 0.5):
    svm_rbf = SVC(C=c_param, probability = True)
    svm_rbf.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_rbf.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

#加多项式核的svm
def svm_poly_module(X, y, indices, c_param, bdry = 0.5):
    svm_poly = SVC(C= c_param[0], kernel = 'poly', degree = c_param[1], probability = True)
    svm_poly.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_poly.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# 逻辑回归
def lr_module(X, y, indices, c_param, bdry = 0.5):
    lr = LogisticRegression(C = c_param, penalty = 'l1')
    # lrcv = LogisticRegressionCV()
    lr.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample= lr.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# 随机森林
def rf_module(X, y, indices, c_param, bdry = 0.5):
    rf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, criterion = 'entropy', max_features = 'auto', max_depth = None, min_samples_split  = c_param, random_state=0)
    rf.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = rf.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# 计算召回率和auc
# y_t 是真实的标签，y_p是预测结果
def compute_recall_and_auc(y_t, y_p):
    # 混淆矩阵
    cnf_matrix = confusion_matrix(y_t, y_p)
    np.set_printoptions(precision=2)
    recall_score = cnf_matrix[1, 1]/(cnf_matrix[1, 0] + cnf_matrix[1, 1])

    # ROC曲线与auc
    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)
    return recall_score, roc_auc

# 很粗暴地用遍历的方式去寻找最优参数组
# 实际上可以用sklearn当中的GridSearchCV写出更优雅的代码
def cross_validation_recall(x_train_data, y_train_data, c_param_range, models_dict, model_name):
    fold = KFold(n_splits=5, shuffle=False)
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    recall_mean = []
    for c_param in c_param_range:
        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data), start=1):

            y_pred_undersample = models_dict[model_name](x_train_data, y_train_data, indices, c_param)

            recall_acc, _ = compute_recall_and_auc(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)

        recall_mean.append(np.mean(recall_accs))

    results_table['Mean recall score'] = recall_mean
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    return best_c

# 不同的决策边界阈值
# 也是通过遍历调参的方式去确定
def decision_boundary(x_train_data, y_train_data, fold,  best_c, bdry_dict, models_dict, model_name):
    bdry_ranges = [0.3, 0.35, 0.4, 0.45, 0.5]
    results_table = pd.DataFrame(index=range(len(bdry_ranges), 2), columns=['C_parameter', 'Mean recall score * auc'])
    results_table['Bdry_params'] = bdry_ranges

    recall_mean = []
    for bdry in bdry_ranges:
        recall_accs_aucs = []
        for iteration, indices in enumerate(fold.split(x_train_data), start=1):
            y_pred_undersample = models_dict[model_name](x_train_data, y_train_data, indices, best_c, bdry)
            recall_acc, roc_auc = compute_recall_and_auc(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs_aucs.append(bdry_dict[model_name](recall_acc, roc_auc))
        recall_mean.append(np.mean(recall_accs_aucs))

    results_table['Mean recall score * auc'] = recall_mean
    best_bdry = results_table.loc[results_table['Mean recall score * auc'].idxmax()]['Bdry_params']

    return best_bdry

# 真正的建模与预测部分
def model(X, y, train, bdry_dict=None, best_c=None, best_bdry=None, models=None, mode=None):
    # 训练阶段
    if train:
        # 用不同的模型
        models_dict = {'knn': knn_module, 'svm_rbf': svm_rbf_module, 'svm_poly': svm_poly_module, 'lr': lr_module, 'rf': rf_module}

        # knn取不同的k值(超参数)
        c_param_range_knn = [3, 5, 7, 9]
        best_c_knn = cross_validation_recall(X, y, c_param_range_knn, models_dict, 'knn')

        #使用GridSearchCV调参
        # parameters = {'C': [ 4, 5], 'degree': [0.01, 0.1]}
        # clf = GridSearchCV(SVC(), parameters,cv=5)
        # clf.fit(X.values,y['Class'].values)
        # print(clf.best_params_)

        # SVM中不同的参数
        c_param_range_svm_rbf = [0.01, 0.1, 1, 10, 100]
        best_c_svm_rbf = cross_validation_recall(X, y, c_param_range_svm_rbf, models_dict, 'svm_rbf')
        c_param_range_svm_poly = [[0.01, 2], [0.01, 3], [0.01, 4], [0.01, 5], [0.01, 6], [0.01, 7], [0.01, 8],[0.01, 9],
                                  [0.1, 2], [0.1, 3], [0.1, 4], [0.1, 5], [0.1, 6], [0.1, 7], [0.1, 8], [0.1, 9],
                                  [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                                  [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9],
                                  [100, 2], [100, 3], [100, 4], [100, 5], [100, 6], [100, 7], [100, 8], [100, 9]]

        best_c_svm_poly = cross_validation_recall(X, y, c_param_range_svm_poly, models_dict, 'svm_poly')

        # 逻辑回归当中的正则化强度
        c_param_range_lr = [0.01, 0.1, 1, 10, 100]
        best_c_lr = cross_validation_recall(X, y, c_param_range_lr, models_dict, 'lr')

        # 随机森林里调参
        c_param_range_rf = [2, 5, 10, 15, 20]
        best_c_rf = cross_validation_recall(X, y, c_param_range_rf, models_dict, 'rf')
        best_c = [best_c_knn, best_c_svm_rbf, best_c_svm_poly, best_c_lr, best_c_rf, best_c]
        # 交叉验证确定合适的决策边界阈值
        # auc 和 召回率存储在bdry_dict之中.
        fold = KFold(4, shuffle=True)

        best_bdry_svm_rbf = decision_boundary(X, y, fold, best_c_svm_rbf, bdry_dict, models_dict, 'svm_rbf')
        best_bdry_svm_poly = decision_boundary(X, y, fold, best_c_svm_poly, bdry_dict, models_dict, 'svm_poly')
        best_bdry_lr = decision_boundary(X, y, fold, best_c_lr, bdry_dict, models_dict, 'lr')
        best_bdry_rf = decision_boundary(X, y, fold, best_c_lr, bdry_dict, models_dict, 'rf')
        best_bdry = [0.5, best_bdry_svm_rbf, best_bdry_svm_poly, best_bdry_lr, best_bdry_rf]

        # 最优参数建模
        knn = KNeighborsClassifier(n_neighbors=int(best_c_knn))
        knn.fit(X.values, y.values.ravel())

        svm_rbf = SVC(C=best_c_svm_rbf, probability=True)
        svm_rbf.fit(X.values, y.values.ravel())

        svm_poly = SVC(C=best_c_svm_poly[0], kernel='poly', degree=best_c_svm_poly[1], probability=True)
        svm_poly.fit(X.values, y.values.ravel())

        lr = LogisticRegression(C=best_c_lr, penalty='l1', warm_start=False)   # 逻辑回归C：惩罚权重系数；penalty:l1正则化

        lr.fit(X.values, y.values.ravel())

        rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, criterion='entropy', max_features='auto',
                                    max_depth=None, min_samples_split=int(best_c_rf), random_state=0)
        rf.fit(X.values, y.values.ravel())

        models = [knn, svm_rbf, svm_poly, lr, rf]

        return best_c, best_bdry, models

    # 预测阶段
    else:
        [knn, svm_rbf, svm_poly, lr, rf] = models
        [_, best_bdry_svm_rbf, best_bdry_svm_poly, best_bdry_lr, best_bdry_rf] = best_bdry

        # KNN
        y_pred_knn = knn.predict(X.values)
        # 用rbf核的SVM
        y_pred_svm_rbf = svm_rbf.predict_proba(X.values)[:, 1] >= best_bdry_svm_rbf
        # 用多项式核的SVM
        y_pred_svm_poly = svm_poly.predict_proba(X.values)[:, 1] >= best_bdry_svm_poly
        # LR
        y_pred_lr = lr.predict_proba(X.values)[:, 1] >= best_bdry_lr
        # 随机森林
        y_pred_rf = rf.predict_proba(X.values)[:, 1] >= best_bdry_rf

        x_of_three_models = {'knn': y_pred_knn, 'svm_rbf': y_pred_svm_rbf, 'svm_poly': y_pred_svm_poly, 'lr': y_pred_lr,
                             'rf': y_pred_rf}
        X_5_data = pd.DataFrame(data=x_of_three_models)   # 汇总5个模型对测试数据预测得到的预测结果

        y_pred = np.sum(X_5_data, axis=1) >= mode   # 5个模型中有>=2个预测为true则为true

        y_pred_lr_controls = []
        params = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        # 投票器去产出最终结果
        for param in params:
            y_pred_lr_controls.append(lr.predict_proba(X.values)[:, 1] >= param)
        return y_pred, y_pred_lr_controls, params

# 总程序
def run(data, mode, ratio, iteration1, bdry_dict):
    recall_score_list = []  # 召回率列表
    auc_list = []   # auc列表
    recall_score_lr_list = []  # 逻辑回归召回率
    auc_lr_list = []  # 逻辑回归auc
    best_c = None
    best_bdry = None
    for itr1 in range(iteration1):
        #print("percentage: %.2f" %(itr1/iteration1*100))

        # 欺诈类的样本
        fraud_indices = np.array(data[data.Class == 1].index)
        np.random.shuffle(fraud_indices)  # 随机重新排序欺诈类样本索引

        # 正常类的样本
        normal_indices = np.array(data[data.Class == 0].index)
        np.random.shuffle(normal_indices)

        # 拆分训练集与测试集
        train_normal_indices, train_fraud_indices, test_normal_indices, test_fraud_indices = split_train_test(
                                                                                            fraud_indices, normal_indices)
        test_indices = np.concatenate([test_normal_indices, test_fraud_indices])  # 将测试的正常类样本索引和欺诈类样本索引链接

        test_data = data.iloc[test_indices, :]   # 根据测试数据的索引获取所有的测试数据
        X_test = test_data.ix[:, test_data.columns != 'Class']
        y_test = test_data.ix[:, test_data.columns == 'Class'].values.ravel()

        # 数据负采样
        X_train_undersample, y_train_undersample, train_normal_pos = getTrainingSample(
                                                                    train_fraud_indices, train_normal_indices, data, 0, ratio)

        # 训练模型
        # 5个模型的最优超参数，最优边界，和模型
        best_c, best_bdry, models = model(X_train_undersample, y_train_undersample, train=True,
                                          bdry_dict=bdry_dict, best_c=best_c, best_bdry=best_bdry)

        if show_best_c:
            print("超参数值:")
            print("k-nearest nbd: %.2f, svm (rbf kernel): [%.2f, %.2f], svm (poly kernel): %.2f, logistic reg: %.2f, random forest: %.2f"
                  % (best_c[0], best_c[1], best_c[2][0], best_c[2][1], best_c[3], best_c[4]))

        if show_bdry:
            print("决策边界阈值:")

            print(
                "k-nearest nbd: %.2f, svm (rbf kernel): %.2f, svm (poly kernel): %.2f, logistic reg: %.2f, random forest: %.2f"
                % (best_bdry[0], best_bdry[1], best_bdry[2], best_bdry[3], best_bdry[4]))

        # 预测
        # 五个模型共同的预测结果，逻辑回归的预测结果，逻辑回归的参数边界组
        y_pred, y_pred_lr_controls, params = model(X_test, y_test, train=False, bdry_dict=None,
                                                       best_c=best_c, best_bdry=best_bdry, models=models, mode=mode)

        # 记录指标
        recall_score, roc_auc = compute_recall_and_auc(y_test, y_pred)
        recall_score_list.append(recall_score)
        auc_list.append(roc_auc)

        control_recall_all_param = []
        control_roc_all_param = []
        for i in range(len(params)):   # 循环逻辑回归的边界值，计算召回率和auc
            recall_score_lr, roc_auc_lr = compute_recall_and_auc(y_test, y_pred_lr_controls[i])  # for control
            control_recall_all_param.append(recall_score_lr)
            control_roc_all_param.append(roc_auc_lr)

        recall_score_lr_list.append(control_recall_all_param)
        auc_lr_list.append(control_roc_all_param)

    # 平均得分
    mean_recall_score = np.mean(recall_score_list)
    std_recall_score = np.std(recall_score_list)

    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    mean_recall_score_lr = np.mean(recall_score_lr_list, axis=0)
    std_recall_score_lr = np.std(recall_score_lr_list, axis=0)
    mean_auc_lr = np.mean(auc_lr_list, axis=0)
    std_auc_lr = np.std(auc_lr_list, axis=0)

    result = [mean_recall_score, std_recall_score, mean_auc, std_auc]
    control = [mean_recall_score_lr, std_recall_score_lr, mean_auc_lr, std_auc_lr]
    return result, control, params


# 一些基本参数设定
mode = 2
ratio = 1
iteration1 = 5  # 迭代次数
show_best_c = True
show_bdry = True

def lr_bdry_module(recall_acc, roc_auc):
    return 0.9*recall_acc+0.1*roc_auc
def svm_rbf_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def svm_poly_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def rf_bdry_module(recall_acc, roc_auc):
    return 0.5*recall_acc+0.5*roc_auc

bdry_dict = {'lr': lr_bdry_module,'svm_rbf': svm_rbf_bdry_module,
             'svm_poly': svm_poly_bdry_module, 'rf': rf_bdry_module}

# 读数据
data = pd.read_csv("D:/materials/dataset/creditcard.csv")
data = data.drop(['Time'], axis=1)

# 对特征做标准化处理
data = normalize_feature(data, amount_only=True)

# 总体开始跑
result, control, params = run(data=data, mode=mode, ratio=ratio, iteration1=iteration1, bdry_dict=bdry_dict)
print("超参数值:")
print("比率为: ", ratio, " 模式为: ", mode)
print("knn, svm_rbf, svm_poly, lr 和 rf 投票产出的结果是:")
print("平均召回率为 ", result[0], " 召回率标准差为 ", result[1])
print("平均auc为 ", result[2], " auc标准差为 ", result[3])
print()
print("调整逻辑回归不同的阈值")
print("我们把超过阈值的样本判定为positive(欺诈)")
for i, param in enumerate(params):
    print("阈值", param)
    print("平均召回率 ", control[0][i], " 召回率标准差 ", control[1][i])
    print("平均auc为 ", control[2][i], " auc标准差 ", control[3][i])
    print()
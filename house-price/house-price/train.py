# -*- coding: utf-8 -*-
"""
Created on Wed June 26 10:00:00 2020
@author: Ashzerm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 导入相关库
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

# 选择模型参数
from sklearn.model_selection import GridSearchCV
# KNNR参数选择
from sklearn.neighbors import KNeighborsRegressor
# RandomForest参数选择
from sklearn.ensemble import RandomForestRegressor

# 导入自定义工具
import Utils.utils as utils

path = './datasets/'

house_price = utils.load_data(path, file_name='house_price.csv')


# -----Step4建模求解-----#


def stat_skew(df, columns=[], n=10, ascending=False):
    # df:需要处理的数据
    # columns:需要处理的变量
    # n:需要显示的偏态变量个数
    # ascending:默认为降序排列
    if columns == []:
        numeric_var = df.dtypes[df.dtypes != "object"].index
    else:
        numeric_var = columns
    # 计算偏态系数时剔除缺失值
    skewed_feats = df[numeric_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=ascending)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness.head(n)
    return skewness


# 自定义函数对右偏态性重的变量进行修正
def revise_skewness(df, skewness, threshold, k):
    # df:需要处理的数据
    # skewness:变量偏态性值数据框
    # threshold:偏态性认证的阈值
    # k:boxcox变换中的lambda参数
    skewness_re = skewness[abs(skewness) > threshold]
    print("There are {} skewed numerical features to Box-Cox transform".format(skewness_re.shape[0]))
    skewed_features = skewness_re.index
    lambda_ = k
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lambda_)
    return df


# 自定义函数进行数据预处理
def transform_max_min(data, columns=[]):
    for col in columns:
        data[col] = (data[col] - min(data[col])) / (max(data[col]) - min(data[col]))
    return data


def SearchK(train_data, train_target, cv=10):
    kf = KFold(n_splits=cv, shuffle=True)  # 数据是否打乱随机
    dic = {}
    for k in (range(5, 16)):
        err_lst = []
        for train_index, test_index in kf.split(train_data):
            x_train, x_test = train_data.loc[train_index], train_data.loc[test_index]
            y_train, y_test = train_target.loc[train_index], train_target.loc[test_index]
            knr = KNeighborsRegressor(n_neighbors=k)
            knr.fit(x_train, y_train)
            y_pre = knr.predict(x_test)
            error = np.sqrt(sum((y_pre - y_test) ** 2) / len(y_test))
            err_lst.append(error)
        dic['Folds%i' % k] = err_lst
    return dic


def line_plot(df):
    plt.figure(figsize=(12, 8))
    for col in df.columns:
        plt.plot(df.index, df[col], '-o', label=col)
    plt.legend()
    plt.title('the Mse of KNNR with K')
    plt.xticks(rotation=1)
    plt.margins(0)
    plt.xlabel('the Serial of Folds')  # X轴标签
    plt.ylabel("Mse")  # Y轴标签


"""
rfr = RandomForestRegressor()
rfr_parameters = {'n_estimators': [500, 800, 1000],
                  'max_depth': [5, 7, 10],
                  'min_samples_leaf': [10],
                  'max_features': ['sqrt'],
                  'bootstrap': [True],
                  'random_state': [2],
                  'max_leaf_nodes': [40, 60, 80]
                  }
rfr_gs = GridSearchCV(estimator=rfr, param_grid=rfr_parameters, cv=5)
rfr_gs.fit(train_data, y_train)
rfr_gs.best_score_
rfr_gs.best_params_
"""


# 设置交互检验折数
k = 10
kf = KFold(n_splits=k, shuffle=True)  # 数据是否打乱随机


# 定义模型训练过程
def train_model(model_name, model, train_data, train_target, test_data):
    accuracies = []
    for train_index, test_index in kf.split(train_data):  # 拆分
        x_train, x_test = train_data.loc[train_index], train_data.loc[test_index]
        y_train, y_test = train_target.loc[train_index], train_target.loc[test_index]
        model.fit(x_train, y_train)  # 训练
        y_predict = model.predict(x_test)  # 预测
        accuracy = np.sqrt(sum((y_predict - y_test) ** 2) / len(y_test))
        accuracies.append(accuracy)
        print('训练完成')
    y_pre = model.predict(test_data)
    print(model_name, np.mean(accuracies), np.std(accuracies))
    return accuracies, y_pre


# 定义模型参数
models = {
    'GBDT': lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, alpha=0.75,
                                              max_depth=7, max_features='sqrt',
                                              min_samples_leaf=10, min_samples_split=20,
                                              loss='huber', random_state=5),
    'Lasso': lambda: Lasso(alpha=0.0005, normalize=True),
    'Knr': lambda: KNeighborsRegressor(n_neighbors=10),
    'Rfr': lambda: RandomForestRegressor(bootstrap=True, max_depth=10, max_features='sqrt',
                                         max_leaf_nodes=80, min_samples_leaf=10,
                                         n_estimators=500, random_state=2)
}


# 模型预测评估
def predict_model(y_test, test_data):
    error_lst = []
    pre_error_lst = []
    for mname, m in models.items():
        error, y_predict = train_model(mname, m(), train_data, y_train, test_data)
        error_lst.append(error)
        pre_error = np.sqrt(sum((y_predict - y_test) ** 2) / len(y_test))
        pre_error_lst.append(pre_error)
        print('{}模型已经训练完成'.format(mname))
    return error_lst, pre_error_lst


def recode_error_lst(models, error_lst):
    dic = {}
    j = 0
    for i in models:
        dic[i] = error_lst[j]
        j += 1
    return dic


if __name__ == "__main__":
    skewness = stat_skew(house_price, columns=[], n=10, ascending=False)
    house_price = revise_skewness(house_price, skewness, 1, 0.15)
    house_price.drop(['id', 'Lng', 'Lat', 'avg_price', 'sta_price'], axis=1, inplace=True)

    total_data = pd.get_dummies(house_price)

    train_data = total_data.iloc[:250000, ]
    test_data = total_data.iloc[250000:, ]

    y_train = train_data['price']
    y_test = test_data['price']
    train_data = train_data.drop(['price'], axis=1)
    test_data = test_data.drop(['price'], axis=1)

    # Klist = SearchK(train_data, y_train)
    print('数据集切分成功')
    error_lst, pre_error_lst = predict_model(y_test, test_data)

    result1 = recode_error_lst(models, error_lst)
    result1 = pd.DataFrame(result1)
    result1.to_csv('./result/result.csv')

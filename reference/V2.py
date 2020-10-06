import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
import xgboost
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from hyperopt import fmin, tpe, hp, partial
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,auc



    def init(train_beh_path,train_tag_path,train_trd_path,pred_beh_path,pred_tag_path,pred_trd_path,):
        train_beh = pd.read_csv(train_beh_path)
        train_beh.index = train_beh.id
        train_beh = train_beh.drop(['id'], axis=1)
        pred_beh = pd.read_csv(pred_beh_path)
        pred_beh.index = pred_beh.id
        pred_beh = pred_beh.drop(['id'], axis=1)
        #check pred id in  train id???  len(set(pred_beh.index))   len(set(train_beh.index)) len(set(pred_beh.index))  len(set(train_beh.index)- set(pred_beh.index))
        # 11913  1844  1232
        beh = pd.concat((train_beh,pred_beh))
        train_tag =  pd.read_csv(train_tag_path)
        train_tag.index = train_tag.id
        train_tag = train_tag.drop(['id'], axis=1)
        train_tag = error_value_deal(train_tag)
        pred_tag = pd.read_csv(pred_tag_path)
        pred_tag.index = pred_tag.id
        pred_tag = pred_tag.drop(['id'], axis=1)
        pred_tag = error_value_deal(pred_tag)
        # check pred id in  train id???  len(set(train_tag.index)) len(set(pred_tag.index))  len(set(train_tag.index)- set(pred_tag.index))
        #39923  6000 4000
        tag = pd.concat((train_tag,pred_tag))
        train_trd = pd.read_csv(train_trd_path)
        train_trd.index = train_trd.id
        train_trd = train_trd.drop(['id'], axis=1)
        pred_trd = pd.read_csv(pred_trd_path)
        pred_trd.index = pred_trd.id
        pred_trd = pred_trd.drop(['id'], axis=1)
        # check pred id in  train id???  len(set(train_trd.index)) len(set(pred_trd.index)) len(set(train_trd.index)- set(pred_trd.index))
        # 31993 4787 3190
        trd = pd.concat((train_trd,pred_trd))
        return beh,tag,trd

    def train_data_run(beh,tag,trd, prt_dt='2019-07-01'):
        beh.isnull().any()
        tag.isnull().any()
        trd.isnull().any()
        # train = pd.merge(tag,beh,on='id')
        # train = pd.merge(train,trd,on='id')
        train = transacation_map(user=tag, transaction=trd, prt_dt=prt_dt)
        train = beh_map(user=train, beh=beh, prt_dt = prt_dt)
        train.isnull().any()
        train['gdr_cd'] = label_hot(train['gdr_cd'])
        train['edu_deg_cd'] = label_hot(train['edu_deg_cd'])
        train['acdm_deg_cd'] = label_hot(train['acdm_deg_cd'])
        train['deg_cd'] = label_hot(train['deg_cd'])
        train['mrg_situ_cd'] = label_hot(train['mrg_situ_cd'])
        for col in train.columns:
            print(col, (train[col] == '\\N').any())
        train = normalization(items=train)
        return train

    def error_value_deal( user_data):
        """
        user_data:用户数据
        mrg_situ_cd
        gdr_cd 性别暂时用Male代替 也可以用第三种性别
        tag 有缺失值的标签 edu_deg_cd教育程度 deg_cd学历  acdm_deg_cd学位  atdd_type信用卡还款方式 出现nan ~ 先把nan转化为~
        deg_cd 缺失较多 考虑drop
        """
        # 使用特征的众数对atdd_type进行填充
        user_data['mrg_situ_cd'] = user_data['mrg_situ_cd'].apply(lambda x: np.nan if x == '\\N' else x)
        user_data['mrg_situ_cd'] = user_data['mrg_situ_cd'].fillna(user_data['mrg_situ_cd'].mode().values[0])  # 填充
        user_data['gdr_cd'] = user_data['gdr_cd'].apply(lambda x: np.nan if x == '\\N' else x)
        user_data['gdr_cd'] = user_data['gdr_cd'].fillna(user_data['gdr_cd'].mode().values[0])  # 填充
        user_data['atdd_type'] = user_data['atdd_type'].apply(lambda x: np.nan if x=='\\N' else x)
        user_data['atdd_type'] = user_data['atdd_type'].fillna(user_data['atdd_type'].median()) #填充
        #(user_data['edu_deg_cd'] == '\\N').any()  (user_data['edu_deg_cd']).unique()
        # (user_data['deg_cd'] == '\\N').any()  (user_data['deg_cd']).unique()
        user_data['edu_deg_cd'] = user_data['edu_deg_cd'].apply(lambda x: np.nan if x == '\\N' else x)
        user_data['edu_deg_cd'] = user_data['edu_deg_cd'].fillna('~')  # 填充
        user_data['deg_cd'] = user_data['deg_cd'].apply(lambda x: np.nan if x == '\\N' else x)
        user_data['deg_cd'] = user_data['deg_cd'].fillna('~')  # 填充
        user_data['acdm_deg_cd'] = user_data['acdm_deg_cd'].apply(lambda x: np.nan if x == '\\N' else x)
        user_data['acdm_deg_cd'] = user_data['acdm_deg_cd'].fillna('~')  # 填充
        for col in (set(user_data.columns)-set({'acdm_deg_cd','deg_cd','atdd_type','edu_deg_cd','gdr_cd','mrg_situ_cd'})) :
            user_data[col] = user_data[col].apply(lambda x: np.nan if x=='\\N' else x)
            user_data[col] = user_data[col].fillna(user_data[col].median()) #填充
        return user_data

    def transacation_map(user, transaction, prt_dt='2019-06-30'):
        """
        k='pred' or 'train'
        transaction:用户交易数据 ['id', 'flag', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd', 'trx_tm', 'cny_trx_amt']
        目标：抽取用户交易数据，确定给定时间到用户交易特征
        """
        print('正在进行' + '交易数据抽取...')
        transaction = transaction[transaction['trx_tm'] <= prt_dt]
        count = transaction['cny_trx_amt'].groupby(transaction.index).count()
        count = count.rename("count")
        amout_mean = transaction['cny_trx_amt'].groupby(transaction.index).mean()
        amout_mean = amout_mean.rename("amout_mean")
        code1_mean = transaction['Trx_Cod1_Cd'].groupby(transaction.index).mean()
        code1_mean = code1_mean.rename("code1_mean")
        code2_mean = transaction['Trx_Cod2_Cd'].groupby(transaction.index).mean()
        code2_mean = code2_mean.rename("code2_mean")
        sz_amt = transaction['cny_trx_amt'].groupby(transaction.index).sum()
        sz_amt = sz_amt.rename("sz_amt")
        temp = (transaction[transaction['cny_trx_amt'] <= 0])['cny_trx_amt']
        expend = temp.groupby(temp.index).sum()
        expend = expend.rename("expend")
        temp = (transaction[transaction['cny_trx_amt'] >= 0])['cny_trx_amt']
        income = temp.groupby(temp.index).sum()
        income = income.rename("income")
        code1 = transaction.groupby([transaction.index, transaction.Trx_Cod1_Cd]).size().unstack(
            fill_value=0)  # 成功把multiindex 做成二维矩阵
        code2 = transaction.groupby([transaction.index, transaction.Trx_Cod2_Cd]).size().unstack(
            fill_value=0)  # 合并user和新造的特征
        user = pd.merge(user, count, how='left', on='id')
        user = pd.merge(user, amout_mean, how='left', on='id')
        user = pd.merge(user, income, how='left', on='id')
        user = pd.merge(user, code1_mean, how='left', on='id')
        user = pd.merge(user, code2_mean, how='left', on='id')
        user = pd.merge(user, sz_amt, how='left', on='id')
        user = pd.merge(user, expend, how='left', on='id')
        user = pd.merge(user, income, how='left', on='id')
        user = pd.merge(user, code1, how='left', on='id')
        user = pd.merge(user, code2, how='left', on='id')
        print('完成' + '交易数据抽取...')
        return user

    def transacation_map(user , transaction , prt_dt='2019-06-30'):
        """
        k='pred' or 'train'
        transaction:用户交易数据 ['id', 'flag', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd', 'trx_tm', 'cny_trx_amt']
        目标：抽取用户交易数据，确定给定时间到用户交易特征
        """
        print('正在进行'+ '交易数据抽取...')
        transaction = transaction[transaction['trx_tm']<= prt_dt]
        sz_amt = transaction['cny_trx_amt'].groupby(transaction.index).sum()
        sz_amt = sz_amt.rename("sz_amt")
        temp = (transaction[ transaction['cny_trx_amt']<=0])['cny_trx_amt']
        expend = temp.groupby(temp.index).sum()
        expend = expend.rename("expend")
        temp = (transaction[transaction['cny_trx_amt'] >= 0])['cny_trx_amt']
        income = temp.groupby(temp.index).sum()
        income = income.rename("income")
        code1 = transaction.groupby([transaction.index,transaction.Trx_Cod1_Cd]).size().unstack(fill_value=0) #成功把multiindex 做成二维矩阵
        code2 = transaction.groupby([transaction.index, transaction.Trx_Cod2_Cd]).size().unstack(fill_value=0) #合并user和新造的特征
        user = pd.merge(user, sz_amt, how='left',on='id')
        user = pd.merge(user, expend, how='left',on='id')
        user = pd.merge(user, income, how='left',on='id')
        user = pd.merge(user,code1, how='left',on='id')
        user = pd.merge(user,code2, how='left',on='id')
        print('完成' + '交易数据抽取...')
        return user

    def beh_map(user , beh , prt_dt='2019-07-01'):
        """
        k='pred' or 'train'
        beh:用户点击app数据 ['id', 'flag', 'page_no', 'page_tm']
        目标：抽取用户交易数据，确定给定时间到用户交易特征
        """
        print('正在进行'+ '用户点击app数据抽取...')
        beh.page_tm =pd.to_datetime(beh.page_tm)
        beh = beh[beh['page_tm']<= prt_dt]

        page_no = beh.groupby([beh.index,beh.page_no]).size().unstack(fill_value=0) #成功把multiindex 做成二维矩阵
        page_tm = beh.page_tm.groupby(beh.index).size() #.unstack(fill_value=0) #合并user和新造的特征
        dayhour = pd.to_datetime(beh.page_tm)
        hour = dayhour.dt.hour.groupby(dayhour.index).mean()
        hour = hour.rename('hour')
        hour_max = dayhour.dt.hour.groupby(dayhour.index).max()
        hour_max = hour_max.rename('hour_max')
        hour_min = dayhour.dt.hour.groupby(dayhour.index).min()
        hour_min = hour_min.rename('hour_min')
        user = pd.merge(user, page_no, how='left',on='id')
        user = pd.merge(user, page_tm, how='left',on='id')
        user = pd.merge(user,hour, how='left',on = 'id')
        user = pd.merge(user,hour_max , how='left',on='id')
        user = pd.merge(user, hour_min, how='left',on='id')
        print('完成' +  '用户点击app数据抽取...')
        return user

    def label_hot(items):
        """
         train  'edu_deg_cd',  'acdm_deg_cd',       'deg_cd',
        对gender进行onehot处理, 由于在之前做过错误值更替处理，所以gender只有M和F两个取值
        """
        le = LabelEncoder()
        print('正在进行' + '数据的label-hot表征...')
        le.fit(items.values)
        le.transform(items)
        print('完成' +  '数据的one-hot表征...')
        return le.transform(items)


    def one_hot(items, k):
        """
        对gender进行onehot处理, 由于在之前做过错误值更替处理，所以gender只有M和F两个取值
        """
        print('正在进行' + k + '数据的one-hot表征...')
        ohe = OneHotEncoder()
        ohe.fit([['M'], ['F']])
        one_hot_data = list(map(lambda x: [x], list(items['gender'])))
        items['gender'] = list(ohe.transform(one_hot_data).toarray())
        joblib.dump(items, res_path + k + '_feature_ohe_data_1.pkl')
        print('完成' + k + '数据的one-hot表征...')
        return items


    def normalization(items):
        """
        items: 将形成的特征向量数据
        对每个用户对特征向量进行归一化处理，使用min_max归一化对方式
        """
        print('正在进行' +  '数据归一化处理...')
        min_max_scaler = MinMaxScaler()
        for col in ( set(items.columns)-set('flag')):
            items[col] = min_max_scaler.fit_transform(items[col].values.reshape(-1, 1)).reshape(1,-1)[0]
        print('完成' + '数据归一化...')
        return items

    def data_prepare(train, oversampling='s', test_size=0.2,seed=1000):
        """
        由于数据集存在不均衡的现象[点击数：未点击数=1:8],所以对训练集数据进行处理，处理方式为对训练集中对点击数据进行随机过采样
        划分训练集和测试集，测试集和训练集分别占比3:7 且按照lable的比例进行抽取
        """
        print('划分测试集和训练集...')
        if oversampling == 's':
            ros = RandomOverSampler(random_state=0)
            new_train_vec, new_train_label = ros.fit_sample(train.loc[:, ~ (data.columns == 'flag')], train.flag)
            train_x, test_x, train_y, test_y = train_test_split(new_train_vec, new_train_label, test_size=test_size,random_state=seed)
        else:
            train_x, test_x, train_y, test_y = train_test_split(train.loc[:, ~ (data.columns == 'flag')], train.flag,
                                                                test_size=test_size, stratify=list(train.flag),random_state=seed)
        return train_x, test_x, train_y, test_y

train_beh_path = 'data/训练数据集/训练数据集_beh.csv'
train_tag_path = 'data/训练数据集/训练数据集_tag.csv'
train_trd_path = 'data/训练数据集/训练数据集_trd.csv'
pred_beh_path = 'data/评分数据集b/评分数据集_beh_b.csv'
pred_tag_path = 'data/评分数据集b/评分数据集_tag_b.csv'
pred_trd_path = 'data/评分数据集b/评分数据集_trd_b.csv'
beh,tag,trd = init(train_beh_path,train_tag_path,train_trd_path,pred_beh_path,pred_tag_path,pred_trd_path,)
data = train_data_run( beh,tag,trd, prt_dt='2019-07-01')


train = data[~data.flag.isnull()]
train.flag = train.flag.astype(int)
train_x, test_x, train_y, test_y = data_prepare(train,oversampling='not',test_size = 0.2,seed = 1000)
#train_x, x_predict,train_y, y_predict = train_test_split(train_x, train_y, test_size=0.10, random_state=100)
pred = data[data.flag.isnull()].loc[:, ~(data.columns =='flag')]


dtrain = xgb.DMatrix(data=train_x,label=train_y,missing=np.nan)
dtest = xgb.DMatrix(data=test_x,label=test_y,missing=np.nan)




# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 15),
         "n_estimators": hp.randint("n_estimators", 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),  #hp.randint("learning_rate", 100),  #
         "subsample": hp.randint("subsample", 5),
         "min_child_weight": hp.randint("min_child_weight", 6),
         }


def xgboost_factory(argsDict=space):
    model = XGBClassifier(booster='gbtree',
                          base_score=0.5,
                          learning_rate=argsDict["learning_rate"] * 0.02 + 0.05,
                          n_estimators=argsDict['n_estimators'] * 10 + 50,  # 树的个数--1000棵树建立xgboost
                          max_depth=8,  # 树的深度
                          min_child_weight=argsDict["min_child_weight"]+1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          reg_alpha=1,  # L1正则化系数，默认为1
                          reg_lambda=0,  # L2正则化系数，默认为1
                          subsample=argsDict["subsample"] * 0.1 + 0.5,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='binary:logistic',  # 指定损失函数
                          scale_pos_weight=0.1,  # 解决样本个数不平衡的问题
                          random_state=1000,  # 随机数
                          missing=np.nan,
                          nthread=-1  # 使用全部CPU进行并行运算（默认）
                          )
    return cross_val_score(model,train_x,train_y.values.ravel(),cv=5,scoring="roc_auc").mean()
    #return auc(y_predict, prediction)

algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(xgboost_factory, space, algo=algo, max_evals=5, pass_expr_memo_ctrl=None)





def fit(argsDict=best):
    model = XGBClassifier(booster='gbtree',
                          base_score=0.5,
                          learning_rate=argsDict["learning_rate"] * 0.02 + 0.05,
                          n_estimators=argsDict['n_estimators'] * 10 + 50,  # 树的个数--1000棵树建立xgboost
                          max_depth=8,  # 树的深度
                          min_child_weight=argsDict["min_child_weight"]+1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          reg_alpha=1,  # L1正则化系数，默认为1
                          reg_lambda=0,  # L2正则化系数，默认为1
                          subsample=argsDict["subsample"] * 0.1 + 0.5,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='binary:logistic',  # 指定损失函数
                          scale_pos_weight=0.1,  # 解决样本个数不平衡的问题
                          random_state=1000,  # 随机数
                          missing=np.nan,
                          nthread=-1  # 使用全部CPU进行并行运算（默认）
                          )
    model.fit(train_x,train_y)
    return model
best_model = fit(best)
result = pd.DataFrame( best_model.predict_proba(pred,),columns=['0','1'],index=pred.index  )
result['id'] = result.index
result[['1']].sort_values(by='id',ascending= True).to_csv( 'data/result.txt', header=None, index=True, sep="\t",encoding='utf-8')





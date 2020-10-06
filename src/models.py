import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import datetime
import logging
logging.basicConfig(format="%(asctime)15s [%(levelname)s]: %(message)s", level=logging.DEBUG)
logger = logging.getLogger()
import pandas as pd
import numpy as np
import  pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
# from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")



def features_sh(data_sh):
    def descrete_var(data, title):  # =data_bj['DOM']
        series = pd.Series(data.values)
        pd.isna(series).any()
        series[pd.isna(series)]
        missing_ratio = sum(pd.isna(series)) / len(series)
        print(title + "变量   missing ratio: %.4f " % (sum(pd.isna(series)) / len(series)))
        # series  = series.astype('int')
        series.describe()
        """
        绘制直方图
        data:必选参数，绘图数据
        bins:直方图的长条形数目，可选项，默认为10
        normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        facecolor:长条形的颜色
        edgecolor:长条形边框的颜色
        alpha:透明度
        """
        plt.cla()
        plt.hist(series, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("区间")
        # 显示纵轴标签
        plt.ylabel("频数/频率")
        # 显示图标题
        plt.title(title + "变量  missing ratio: %.4f " % (sum(pd.isna(series)) / len(series)))
        plt.savefig('./pictures/sh/' + title + '.jpg')
        plt.show()
        return missing_ratio

    # 特征工程
    dic = {'嘉定':2,  '奉贤':2,   '宝山':2,    '崇明':3,    '徐汇':1,    '普陀':1,    '杨浦':1,    '松江':2,    '浦东':2,
    '虹口':1,    '金山':3,    '长宁':1,    '闵行':2,    '闸北':1,    '青浦':2,    '静安':1,    '黄浦':1}
    data_sh['district'] = data_sh['district'].apply(
        lambda s:dic[s] if not pd.isna(s) else np.nan)
    dic = {'中楼层': 2, '低楼层': 1, '底楼层': 1, '(共0层': np.nan,  '顶楼层': 4, '高楼层': 3}
    data_sh['floor'] = data_sh['floor'].apply(
        lambda s: dic[s] if not pd.isna(s) and len(s)==3 else np.nan)
    dic = {'塔楼': 1, '平房': 2, '板塔': 3, '板楼': 4, '暂无':np.nan,'':np.nan}
    data_sh['buildingType'] = data_sh['buildingType'].apply(
        lambda s: dic[s] if not pd.isna(s) else np.nan)
    dic = {'其他': 1, '毛坯': 2, '简装': 3, '精装': 4, }
    data_sh['renovationCondition'] = data_sh['renovationCondition'].apply(
        lambda s: dic[s] if not pd.isna(s) and len(s)==2 else np.nan)
    dic = {'未知结构': 1, '混合结构': 2, '砖木结构': 3, '砖混结构': 4, '钢结构':5, '钢混结构':6}
    data_sh['buildingStructure'] = data_sh['buildingStructure'].apply(
        lambda s: dic[s] if not pd.isna(s) and len(str(s))>1 else np.nan)
    dic = {'有': 1, '无': 0,'据':1 }
    data_sh['elevator'] = data_sh['elevator'].apply(
        lambda s: dic[s] if not pd.isna(s) else np.nan)
    dic = {'满五年': 1, '暂无数据': 0,'满两年': 1, '未满两年': -1}
    data_sh['fiveYearsProperty'] = data_sh['fiveYearsProperty'].apply(
        lambda s: dic[s] if not pd.isna(s) and len(s) >1 else np.nan)


    keep_features = [ 'price', 'totalPrice', 'first_installment', 'constructionTime',
       'district',  'Lng', 'Lat', 'livingRoom',
       'drawingRoom', 'kitchen', 'bathRoom', 'floor', 'floor_num', 'square',
       'unitStructure', 'buildingType', 'orient',
       'buildingStructure', 'renovationCondition', 'ladderRatio', 'elevator',
        'onBoard year', 'onBoard month', 'trading_property',
       'use_property', 'fiveYearsProperty', 'owner',
       'mortgage', 'update_certificate']
    data_sh  =  data_sh[keep_features]

    # 清洗异常值
    '''
    暂时没有
    '''
    missing = pd.DataFrame({'whether missing': pd.isna(data_sh).any().values, 'missing ratio': np.nan},
                           index=pd.isna(data_sh).any().index)

    for col in data_sh.columns:  # data_bj.columns
        try:
            # descrete_var( data_bj[col],title = col)
            # time.sleep(5)
            missing.loc[col, 'missing ratio'] = descrete_var(data_sh[col], col)
        except:
            pass
    # missing.to_csv('missing.csv')
    # 清洗空缺值  pd.isna(data_sh['price']).any()
    data_sh = data_sh   [   ~pd.isna(data_sh['price'])  ]
    squ_bins = [-1, 20, 50, 90, 120, 160, 200, 2000]
    data_sh['squ_level'] = pd.cut(data_sh['square'], bins=squ_bins,
                                  labels=['20平以下', '20-50', '50-90', '90-120', '120-160', '160-200', '200平以上'])
    data_sh['squ_level'] = data_sh['squ_level'].fillna('50-90')
  #  for col in cols:
  #      col = 'livingRoom'
  #      for label in ['20平以下', '20-50', '50-90', '90-120', '120-160', '160-200', '200平以上']:
  #          data_sh[col][data_sh['squ_level'] == label ] = data_sh[col][data_sh['squ_level'] == label ].fillna(   (data_sh[col][data_sh['squ_level'] == label ]) .median()     )

    data_sh['livingRoom'] = data_sh.groupby('squ_level')['livingRoom'].transform(lambda x: x.fillna(x.median()))
    data_sh['drawingRoom'] = data_sh.groupby('squ_level')['drawingRoom'].transform(lambda x: x.fillna(x.median()))
    data_sh['kitchen'] = data_sh.groupby('squ_level')['kitchen'].transform(lambda x: x.fillna(x.median()))
    data_sh['bathRoom'] = data_sh.groupby('squ_level')['bathRoom'].transform(lambda x: x.fillna(x.mean()))
    data_sh['floor'] = data_sh.groupby('squ_level')['floor'].transform(lambda x: x.fillna(x.median()));
#    data_sh['floor'] = data_sh['floor'].fillna(data_sh['floor'].median())
    data_sh['totalPrice'] = data_sh.groupby('squ_level')['totalPrice'].transform(lambda x: x.fillna(x.median()))
    data_sh['floor_num'] = data_sh.groupby('squ_level')['floor_num'].transform(lambda x: x.fillna(x.median()))
    data_sh['buildingType'] = data_sh.groupby('squ_level')['buildingType'].transform(lambda x: x.fillna(x.median()))
    data_sh['elevator'] =data_sh.groupby('squ_level')['elevator'].transform(lambda x: x.fillna(x.median()))
    data_sh['fiveYearsProperty'] = data_sh.groupby('squ_level')['fiveYearsProperty'].transform(lambda x: x.fillna(x.median()))
    data_sh['ladderRatio'] = data_sh.groupby('squ_level')['ladderRatio'].transform(lambda x: x.fillna(x.median()))
    data_sh['constructionTime'] = data_sh.groupby('squ_level')['constructionTime'].transform(lambda x: x.fillna(x.median()))
    data_sh['onBoard year'] = data_sh.groupby('squ_level')['onBoard year'].transform(lambda x: x.fillna(x.median()))
    data_sh['onBoard month'] = data_sh.groupby('squ_level')['onBoard month'].transform(lambda x: x.fillna(x.median()))
    data_sh['first_installment'] = data_sh.groupby('squ_level')['first_installment'].transform(lambda x: x.fillna(x.median()))
    data_sh['district'] = data_sh.groupby('squ_level')['district'].transform(lambda x: x.fillna(x.median()))
    data_sh['Lat'] = data_sh.groupby('squ_level')['Lat'].transform(lambda x: x.fillna(x.median()))
    data_sh['Lng'] = data_sh.groupby('squ_level')['Lng'].transform(lambda x: x.fillna(x.median()))
    data_sh['buildingStructure'] = data_sh.groupby('squ_level')['buildingStructure'].transform(lambda x: x.fillna(x.median()))
    data_sh['renovationCondition'] = data_sh.groupby('squ_level')['renovationCondition'].transform(lambda x: x.fillna(x.median()))
    data_sh['square'] = data_sh.groupby('squ_level')['square'].transform(lambda x: x.fillna(x.median()))

    data_sh['mortgage'] = data_sh['mortgage'].fillna("暂无数据")
    data_sh['unitStructure'] = data_sh['unitStructure'].fillna("暂无")
    data_sh['orient'] = data_sh['orient'].fillna( "未知")
    data_sh['update_certificate'] = data_sh['update_certificate'].fillna( "未知")
    data_sh['use_property'] = data_sh['use_property'].fillna("暂无数据")
    data_sh['trading_property'] = data_sh['trading_property'].fillna("")
    data_sh['owner'] = data_sh['owner'].fillna("暂无数据")
    #pd.Series( list( set(data_sh['trading_property'])  )).dropna()
    assert  not pd.isna(  (data_sh)  ).any().any()

    '''
        label-hot   ['unitStructure']
        '''
    from sklearn.preprocessing import LabelEncoder
    for col in ['mortgage', 'unitStructure','orient','update_certificate','use_property','trading_property','owner']:
        data_sh[col] = LabelEncoder().fit(data_sh[col]).transform(data_sh[col])

    # 强制类型转化
    # types = dict( zip( data_bj.dtypes.index, list( map(lambda x:str(x), data_bj.dtypes.values) )  ))
    data_sh = data_sh.drop(['squ_level'],axis=1)
    types = {'Lng': 'float64',
             'Lat': 'float64',
             'totalPrice': 'int64',
             'price': 'int64',
             'first_installment':'int64',
             'square': 'int64',
             'livingRoom': 'int64',
             'drawingRoom': 'int64',
             'kitchen': 'int64',
             'bathRoom': 'int64',
             'floor': 'int64',
             'buildingType': 'int64',
             'unitStructure':'int64',
             'orient':'int64',
             'ladderRatio':'float64',
             'onBoard year':'int64',
             'onBoard month':'int64',
             'mortgage':'int64',
             'update_certificate':'int64',
             'use_property':'int64',
             'trading_property':'int64',
             'owner':'int64',
             'constructionTime': 'int64',
             'renovationCondition': 'int64',
             'buildingStructure': 'int64',
             'ladderRatio': 'float64',
             'elevator': 'int64',
             'fiveYearsProperty': 'int64',
             'subway': 'int64',
             'district': 'int64',
             'communityAverage': 'int64',
             'floor_num': 'int64'}
    for col in data_sh.columns:
        data_sh[col] = data_sh[col].astype(types[col])

    return data_sh



'''
['id', 'Lng', 'Lat', 'Cid', 'tradeTime', 'DOM', 'followers',
       'totalPrice', 'price', 'square', 'livingRoom', 'drawingRoom', 'kitchen',
       'bathRoom', 'floor', 'buildingType', 'constructionTime',
       'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
       'fiveYearsProperty', 'subway', 'district', 'communityAverage',
       'floor_num']
'''
def features_bj(data_bj):
    def descrete_var(data, title):  # =data_bj['DOM']
        series = pd.Series(data.values)
        pd.isna(series).any()
        series[pd.isna(series)]
        missing_ratio = sum(pd.isna(series)) / len(series)
        print(title + "变量   missing ratio: %.4f " % (sum(pd.isna(series)) / len(series)))
        # series  = series.astype('int')
        series.describe()
        """
        绘制直方图
        data:必选参数，绘图数据
        bins:直方图的长条形数目，可选项，默认为10
        normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        facecolor:长条形的颜色
        edgecolor:长条形边框的颜色
        alpha:透明度
        """
        plt.cla()
        plt.hist(series, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("区间")
        # 显示纵轴标签
        plt.ylabel("频数/频率")
        # 显示图标题
        plt.title(title + "变量  missing ratio: %.4f " % (sum(pd.isna(series)) / len(series)))
        plt.savefig('./pictures/bj/' + title + '.jpg')
        plt.show()
        return missing_ratio
    # 清洗异常值
    dic = {'中': 2, '低': 1, '底': 1, '未知': np.nan, '混合结构': np.nan, '钢混结构': np.nan, '顶': 4, '高': 3}
    # set(data_bj['floor'])
    data_bj['floor'] = data_bj['floor'].apply(lambda x: dic[x], )
    data_bj['floor_num'] = data_bj['floor_num'].apply(lambda x: int(x) if len(str(x)) == 1 else np.nan, )
    data_bj['livingRoom'] = data_bj['livingRoom'].apply(lambda x: int(x) if x != '#NAME?' else np.nan, )
    # set(data_bj['floor_num'])
    data_bj['drawingRoom'] = data_bj['drawingRoom'].apply(lambda x: int(x) if len(str(x)) == 1 else np.nan, )
    data_bj['bathRoom'] = data_bj['bathRoom'].apply(lambda x: int(x) if len(str(x)) == 1 else np.nan, )
    data_bj['constructionTime'] = data_bj['constructionTime'].apply(lambda x: int(x) if len(str(x)) == 4 else np.nan, )

    missing = pd.DataFrame({'whether missing': pd.isna(data_bj).any().values, 'missing ratio': np.nan},
                           index=pd.isna(data_bj).any().index)

    for col in data_bj.columns:  # data_bj.columns
        try:
            # descrete_var( data_bj[col],title = col)
            # time.sleep(5)
            missing.loc[col, 'missing ratio'] = descrete_var(data_bj[col], col)
        except:
            pass
    # missing.to_csv('missing.csv')
    # 清洗空缺值  pd.isna(data_bj['constructionTime']).any()
    squ_bins = [-1, 20, 50, 90, 120, 160, 200, 2000]
    data_bj['squ_level'] = pd.cut(data_bj['square'], bins=squ_bins,
                                  labels=['20平以下', '20-50', '50-90', '90-120', '120-160', '160-200', '200平以上'])
    data_bj['livingRoom'] = data_bj.groupby('squ_level')['livingRoom'].transform(lambda x: x.fillna(x.median()))
    data_bj['drawingRoom'] = data_bj.groupby('squ_level')['drawingRoom'].transform(lambda x: x.fillna(x.median()))
    data_bj['kitchen'] = data_bj.groupby('squ_level')['kitchen'].transform(lambda x: x.fillna(x.median()))
    data_bj['bathRoom'] = data_bj.groupby('squ_level')['bathRoom'].transform(lambda x: x.fillna(x.median()))
    data_bj['floor_num'] = data_bj.groupby('squ_level')['floor_num'].transform(lambda x: x.fillna(x.median()))
    data_bj['totalPrice'] = data_bj.groupby('squ_level')['totalPrice'].transform(lambda x: x.fillna(x.median()))
    data_bj['floor_num'] = data_bj.groupby('squ_level')['floor_num'].transform(lambda x: x.fillna(x.median()))
    data_bj['floor'] = data_bj.groupby('squ_level')['floor'].transform(lambda x: x.fillna(x.median()))
    data_bj['DOM'] = data_bj.groupby('squ_level')['DOM'].transform(lambda x: x.fillna(x.median()))
    data_bj['buildingType'] = data_bj.groupby('squ_level')['buildingType'].transform(lambda x: x.fillna(x.median()))
    data_bj['elevator'] = data_bj.groupby('squ_level')['elevator'].transform(lambda x: x.fillna(x.median()))
    data_bj['fiveYearsProperty'] = data_bj.groupby('squ_level')['fiveYearsProperty'].transform(
        lambda x: x.fillna(x.median()))
    data_bj['communityAverage'] = data_bj.groupby('squ_level')['communityAverage'].transform(
        lambda x: x.fillna(x.median()))
    data_bj['subway'] = data_bj.groupby('squ_level')['subway'].transform(lambda x: x.fillna(x.median()))
    data_bj['constructionTime'] = data_bj.groupby('squ_level')['constructionTime'].transform(
        lambda x: x.fillna(x.median()))

    # 特征工程
    data_bj['tradeTime_year'] = data_bj['tradeTime'].apply(
        lambda s: pd.to_datetime(s).year if not pd.isna(pd.to_datetime(s).year) else np.nan)
    data_bj['tradeTime_month'] = data_bj['tradeTime'].apply(
        lambda s: pd.to_datetime(s).month if not pd.isna(pd.to_datetime(s).month) else np.nan)

    features = ['Lng', 'Lat', 'Cid', 'tradeTime_year', 'tradeTime_month', 'DOM', 'followers',
                'totalPrice', 'price', 'square', 'livingRoom', 'drawingRoom', 'kitchen',
                'bathRoom', 'floor', 'buildingType', 'constructionTime',
                'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
                'fiveYearsProperty', 'subway', 'district', 'communityAverage',
                'floor_num']
    data_bj = data_bj[features]

    # 强制类型转化
    assert not pd.isna(data_bj).any().any()
    # types = dict( zip( data_bj.dtypes.index, list( map(lambda x:str(x), data_bj.dtypes.values) )  ))
    types = {'Lng': 'float64',
             'Lat': 'float64',
             'Cid': 'int64',
             'tradeTime_year': 'int64',
             'tradeTime_month': 'int64',
             'DOM': 'int64',
             'followers': 'int64',
             'totalPrice': 'int64',
             'price': 'int64',
             'square': 'int64',
             'livingRoom': 'int64',
             'drawingRoom': 'int64',
             'kitchen': 'int64',
             'bathRoom': 'int64',
             'floor': 'int64',
             'buildingType': 'int64',
             'constructionTime': 'int64',
             'renovationCondition': 'int64',
             'buildingStructure': 'int64',
             'ladderRatio': 'float64',
             'elevator': 'int64',
             'fiveYearsProperty': 'int64',
             'subway': 'int64',
             'district': 'int64',
             'communityAverage': 'int64',
             'floor_num': 'int64'}
    for col in data_bj.columns:
        data_bj[col] = data_bj[col].astype(types[col])

    return data_bj


class Machine_learning(object):
    def __init__(self, data, flag):
        self.data = data
        self.flag = flag
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    def data_prepare(self, oversampling, test_size, seed):
        """
        如果未来考虑 训练集 验证集 测试集
        #pred = data[data.flag.isnull()].loc[:, ~(data.columns == flag)]
        #pred = pred.fillna(0)
        """
        logger.info('划分测试集和训练集...')
        if oversampling == 's':
            ros = RandomOverSampler(random_state=0)
            new_train_vec, new_train_label = ros.fit_sample(self.data.loc[:, ~ (self.data.columns == self.flag)], self.data[self.flag])
            train_x, test_x, train_y, test_y = train_test_split(new_train_vec, new_train_label, test_size=test_size,
                                                                random_state=seed)
        else:
            train_x, test_x, train_y, test_y = train_test_split(self.data.loc[:, ~ (self.data.columns == self.flag)], self.data[self.flag],
                                                                test_size=test_size, random_state=seed)
        logger.info('划分测试集和训练集完毕')
        return train_x, test_x, train_y, test_y

    def lgb_fit(self):
        lgb_train = lgb.Dataset(data=self.X_train, label=self.y_train, free_raw_data=False)
        lgb_test = lgb.Dataset(data=self.X_test, label=self.y_test, reference=lgb_train, free_raw_data=False)
        # https://github.com/albertkklam/LGBMRegressor/blob/master/LGBMRegressor.ipynb
        params = {'boosting_type': 'dart',
                  'objective': 'regression',
                  'metric': 'l2',
                  'num_leaves': 10,
                  'max_depth': -1,
                  'learning_rate': 0.02,
                  'n_estimators': 1000,
                  'min_split_gain': 0.05,
                  'min_child_weight': 0.5,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'reg_alpha': 0.2,
                  'reg_lambda': 0.2,
                  'drop_rate': 0.2,
                  'skip_drop': 0.8,
                  'max_drop': 200,
                  'seed': 100,
                  'silent': False
                  }
        gbmCV = lgb.cv(params,
                       train_set=lgb_train,
                       num_boost_round=1000,
                       nfold=5,
                       early_stopping_rounds=10,
                       verbose_eval=True
                       )
        logger.info("Cross validation completed!")
        logger.info("num_boost_round",gbmCV['l2-mean'])
        num_boost_round = len(gbmCV['l2-mean'])
        gbm = lgb.train(params,
                        train_set=lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=[lgb_test],
                        valid_names=['eval'],
                        evals_result={},
                        verbose_eval=True
                        )
        logger.info("Booster object completed!")
        lgb.plot_importance(gbm, max_num_features=30, importance_type='split')
        plt.show()
        importance = pd.DataFrame()
        importance['Feature'] = self.X_train.columns.values
        importance['ImportanceWeight'] = gbm.feature_importance(importance_type='split')
        importance['ImportanceGain'] = gbm.feature_importance(importance_type='gain')
        importance.sort_values(by='ImportanceWeight', ascending=False, inplace=True)
        importance.head()
        plt.savefig("./pictures/feature_importance/" + type(gbm).__name__ + ".jpg")
        return gbm

    def xgb_fit(self):
        parameters = {
            'objective': 'reg:linear',
            'seed': 100,
            'learning_rate': 0.01,
            'n_estimators': 10,
            'silent': False,
            'n_jobs': 4,
            'nthread': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            #    'reg_alpha' : 10,  # L1 regularization term on weights. Increasing this value will make model more conservative.
            #    'reg_lambda' : 1,  # L2 regularization term on weights. Increasing this value will make model more conservative.
            # base_score=0.5, #The initial prediction score of all instances, global bias
            # 'updater':'shotgun',  # Parallel coordinate descent algorithm based on shotgun algorithm.
            # 'importance_type': "gain",
        }
        cv_params = {  # 'max_depth': np.linspace(2,10,5),
            # 'gamma':np.linspace(0.1,1,10),
            'reg_lambda': np.linspace(0.1, 10, 2),
            'reg_alpha': np.linspace(0.1, 10, 2)
        }
        gbm = GridSearchCV(
            xgb.XGBRegressor(
                objective="reg:linear",
                seed=100,
                gamma=1,
                n_estimators=100,
                max_depth=3,
                min_child_weight=5,
                learning_rate=0.01,
                colsample_bytree=0.8,
                subsample=0.8,
                silent=False
            ),
            param_grid=cv_params,
            n_jobs=4,
            iid=False,
            scoring="neg_mean_squared_error",
            cv=5,
            verbose=True
        )
        gbm.fit(self.X_train, self.y_train)
        print("Best parameters %s" % gbm.best_params_)
        for key, value in gbm.best_params_.items():
            parameters[key] = value
        trainDMat = xgb.DMatrix(data = self.X_train, label = self.y_train)
        testDMat = xgb.DMatrix(data = self.X_test, label = self.y_test)
        xgbCV = xgb.cv(
            params=parameters,
            dtrain=trainDMat,
            num_boost_round=1000,
            nfold=5,
            metrics={'rmse'},
            early_stopping_rounds=500,
            verbose_eval=True,
            seed=0
        )

        logger.info("Training complete! Producing final booster object")
        parameters['eval_metric'] = 'rmse'
        xgbFinal = xgb.train(
            params=parameters,
            dtrain=trainDMat,
            num_boost_round=1000,
            evals=[(trainDMat, 'train'),
                   (testDMat, 'eval')]
        )
        logger.info("Booster object created!")
        plot_importance(xgbFinal)
        plt.show()
        plt.savefig("./pictures/feature_importance/" + type(xgbFinal).__name__ + " xgb"+ ".jpg")
        return xgbFinal


    def do_fit(self, model):
        model.fit(self.X_train, self.y_train)
        return model

    def do_predict(self,model):
        model.predict(self.X_test)
        # 获得这个模型的参数
        #model.get_params()
        # 为模型进行打分
        score = model.score(self.X_test, self.y_test)  # 线性回归：R square； 分类问题： acc
        print(type(model).__name__," score: ",score)
        return score

    def plot_fit(self,model):
        y_prediction = model.predict(self.X_test)
        score = r2_score(self.y_test, y_prediction)
        plt.cla()
        plt.scatter(self.y_test, y_prediction)
        plt.xlabel("y_test")
        plt.ylabel("y_prediction")
        plt.title(type(model).__name__ + "  with r2 score: " +str(score))
        plt.show()
        plt.savefig("./pictures/plot_fit/"+ "plot_fit_of_ "  +  type(model).__name__ +  ".jpg")
    def plot_feature_importance(self,model):
        print("type(model).__name__"," model.alpha_ : ", model.alpha_)
        coef = pd.Series(model.coef_, index=self.X_train.columns)  # .coef_ 可以返回经过学习后的所有 feature 的参数。
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
            sum(coef == 0)) + " variables")
        imp_coef = pd.concat([coef.sort_values().head(10),
                              coef.sort_values().tail(10)])  # 选头尾各10条，.sort_values() 可以将某一列的值进行排序。
        plt.cla()
        imp_coef.plot(kind="barh");
        plt.title("Coefficients in the "  + type(model).__name__ + " Model")
        plt.show()
        plt.savefig("./pictures/feature_importance/" +   type(model).__name__ + ".jpg")

    def run(self):


        res  = {}
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(oversampling='not', test_size=0.2,seed=1000)
        logger.info("#------ 定义线性回归模型-------#")
        model_linear = LinearRegression(fit_intercept=True, normalize=False,copy_X=True, n_jobs=1)
        logger.info("#------ fit 线性回归模型-------#")
        model_linear = self.do_fit(model_linear)
        logger.info("#------ predict 线性回归模型-------#")
        model_linear_score = self.do_predict(model_linear)
        self.plot_fit(model_linear)
        res['LinearRegression'] = model_linear_score

        logger.info("#------ fit xgb模型-------#")
        xgb = self.xgb_fit()
        logger.info("#------ predict xgb模型-------#")
        xgb_score = r2_score(xgb.predict(self.X_test), self.y_test)
        self.plot_fit(xgb)
        res['xgb'] = xgb_score

        logger.info("#------ 定义lasso模型-------#")
        model_lasso = LassoCV(alphas=np.linspace(0.1, 1000, 100))
        logger.info("#------ fit lasso模型-------#")
        model_lasso = self.do_fit(model_lasso)
        logger.info("#------ predict lasso模型-------#")
        model_lasso_score = self.do_predict(model_lasso)
        self.plot_feature_importance(model_lasso)
        self.plot_fit(model_lasso)
        res['lasso'] = model_lasso_score


        logger.info("#------ 定义弹性网络模型-------#")
        # https://blog.csdn.net/previous_moon/article/details/71376726
        # 1 / (2 * n_samples) * ||y - Xw||^2_2+ alpha * l1_ratio * ||w||_1+ 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
        model_ElasticNet = ElasticNetCV(alphas=np.linspace(0.1,10,10), l1_ratio=np.linspace(0.0,1,11),  max_iter=500)
        logger.info("#------ fit 弹性网络模型-------#")
        model_ElasticNet = self.do_fit(model_ElasticNet)
        logger.info("#------ predict 弹性网络模型-------#")
        model_ElasticNet_score = self.do_predict(model_ElasticNet)
        self.plot_feature_importance(model_ElasticNet)
        self.plot_fit(model_linear)
        res['ElasticNet'] = model_ElasticNet_score
        
        logger.info("#------ fit lgb模型-------#")
        gbm = self.lgb_fit()
        logger.info("#------ predict lgb模型-------#")
        gbm_score = r2_score( gbm.predict(self.X_test), self.y_test)
        self.plot_fit(gbm)
        res['lgb'] =  gbm_score

        res = pd.DataFrame({'model_name': list(res.keys()), 'r2_score': list(res.values())})
        res.to_csv('./result/train_result.csv')
        return gbm  #将来考虑这里输出best_model




 def predict_2018(model,data,s):
    if(s=='bj'):
        data_2018 = data[ (data.tradeTime_year)==2018]
    else:
        data_2018 = data[(data['onBoard year']) == 2018]
    flag  = 'price'
    X_pred, y_pred = data_2018.loc[:, ~ (data_2018.columns == flag)], data_2018[flag]
    y_prediction = model.predict(X_pred)
    score = r2_score(y_prediction,y_pred)
    print(score)
    plt.cla()
    plt.scatter(y_pred, y_prediction)
    plt.xlabel("y_pred")
    plt.ylabel("y_prediction")
    plt.title("2018" + type(model).__name__ + "  with r2 score: " +str(score))
    plt.show()
    plt.savefig("./pictures/plot_fit/"+ "2018_plot_fit_of_ "  +  type(model).__name__ +  ".jpg")

def plot_avg_price(data_sh,data_bj):
    plt.cla()
    price_bj = data_bj.groupby('tradeTime_year')['price'].mean()
    price_sh = data_sh.groupby('onBoard year')['price'].mean()
    price = pd.DataFrame({'price_bj': price_bj,'price_sh':price_sh })
    price.plot()
    #plt.plot(price_bj.index, price_bj.values, label="migration time",color = 'r')
    #plt.plot(price_bj.index, ([np.nan]*6).extend(price_sh.values) , label="request delay",color = 'b')
    """open the grid"""
    plt.grid(True)
    plt.title("house avg price")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


if __name__ == "main":

    #北京
    data_bj = pickle.load(open('./data/data_before_map_bj.pkl', 'rb'))
    data_bj = features_bj(data_bj)
    regression = Machine_learning(data_bj, 'price')
    model = regression.run()
    predict_2018(model, data_bj,'bj')

    #去除totalPrice square CommunityAverage
    reserve = ['Lng', 'Lat', 'Cid', 'tradeTime_year', 'tradeTime_month', 'DOM',
       'followers',   'livingRoom','price',
       'drawingRoom', 'kitchen', 'bathRoom', 'floor', 'buildingType',
       'constructionTime', 'renovationCondition', 'buildingStructure',
       'ladderRatio', 'elevator', 'fiveYearsProperty', 'subway', 'district', 'floor_num']
    data_bj_new = data_bj[reserve]

    regression_new = Machine_learning(data_bj_new, 'price')
    model_new = regression.run()
    predict_2018(model_new, data_bj_new, 'bj')


    #上海
    data_sh = pickle.load(open('./data/data_before_map_sh.pkl', 'rb'))
    data_sh = features_sh(data_sh)
    regression = Machine_learning(data_sh, 'price')
    model_sh = regression.run()
    predict_2018(model_sh,data_sh,'sh')
    #train = data[~data[flag].isnull()]
    #train[flag] = train[flag].astype(int)
    #train = train.fillna(0)
    #flag = 'price'
    #X_train, X_test, y_train, y_test = train_test_split(data.loc[:, ~ (data.columns == flag)], data[flag], test_size=0.3, random_state=42)


    # 去除totalPrice square CommunityAverage
    reserve =['price',  'first_installment', 'constructionTime',
       'district', 'Lng', 'Lat', 'livingRoom', 'drawingRoom', 'kitchen',
       'bathRoom', 'floor', 'floor_num', 'square', 'unitStructure',
       'buildingType', 'orient', 'buildingStructure', 'renovationCondition',
       'ladderRatio', 'elevator', 'onBoard year', 'onBoard month',
       'trading_property', 'use_property', 'fiveYearsProperty', 'owner',
       'mortgage', 'update_certificate']

    data_sh_new = data_sh[reserve]
    regression = Machine_learning(data_sh_new, 'price')
    model_sh_new  = regression.run()

    #画出北京上海房价均价
    plot_avg_price(data_sh, data_bj)


'''
data_sh = pickle.load(open('./data/data_before_map_sh.pkl', 'rb'))
data = features_sh(data_sh)
flag = 'price'
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, ~ (data.columns == flag)], data[flag], test_size=0.3, random_state=42)

parameters = {
    'objective': 'reg:linear',
    'seed':100,
    'learning_rate':0.01,
    'n_estimators': 10,
    'silent': False,
    'n_jobs': 4,
    'nthread' : -1,
    'subsample':0.8,
    'colsample_bytree':0.8,
#    'reg_alpha' : 10,  # L1 regularization term on weights. Increasing this value will make model more conservative.
#    'reg_lambda' : 1,  # L2 regularization term on weights. Increasing this value will make model more conservative.
    # base_score=0.5, #The initial prediction score of all instances, global bias
    #'updater':'shotgun',  # Parallel coordinate descent algorithm based on shotgun algorithm.
    #'importance_type': "gain",
}
cv_params = {#'max_depth': np.linspace(2,10,5),
             #'gamma':np.linspace(0.1,1,10),
             'reg_lambda': np.linspace(0,10,1),
             'reg_alpha':np.linspace(0,10,1)
            }
gbm = GridSearchCV(
        xgb.XGBRegressor(
            objective ="reg:linear",
            seed = 100,
            gamma= 1,
            n_estimators = 100,
            max_depth =3,
            min_child_weight = 5,
            learning_rate = 0.01,
            colsample_bytree = 0.8,
            subsample = 0.8,
            silent = False
        ),
        param_grid=cv_params,
        n_jobs = 4,
        iid=False,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=True
)
gbm.fit(X_train, y_train)
print("Best parameters %s" %gbm.best_params_)
for key,value in gbm.best_params_.items():
    parameters[key] = value



trainDMat = xgb.DMatrix(data = X_train, label = y_train)
testDMat = xgb.DMatrix(data = X_test, label = y_test)
xgbCV = xgb.cv(
    params = parameters,
    dtrain = trainDMat,
    num_boost_round = 1000,
    nfold = 5,
    metrics = {'rmse'},
    early_stopping_rounds = 500,
    verbose_eval = True,
    seed = 0
)

logger.info("Training complete! Producing final booster object")
parameters['eval_metric'] = 'rmse'
xgbFinal = xgb.train(
    params = parameters,
    dtrain = trainDMat,
    num_boost_round = 1000,
    evals = [(trainDMat, 'train'),
             (testDMat, 'eval')]
)
logger.info("Booster object created!")
plot_importance(xgbFinal)
plt.show()

y_preds = xgbFinal.predict(testDMat )
r2_score(y_preds,y_test)



def label_hot(items):
    """
     train  'edu_deg_cd',  'acdm_deg_cd',       'deg_cd',
    对gender进行onehot处理, 由于在之前做过错误值更替处理，所以gender只有M和F两个取值
    """
    le = LabelEncoder()
    print('正在进行' + '数据的label-hot表征...')
    le.fit(items.values)
    le.transform(items)
    print('完成' + '数据的label-hot表征...')
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
    print('正在进行' + '数据归一化处理...')
    min_max_scaler = MinMaxScaler()
    for col in (set(items.columns) - set('flag')):
        items[col] = min_max_scaler.fit_transform(items[col].values.reshape(-1, 1)).reshape(1, -1)[0]
    print('完成' + '数据归一化...')
    return items


def data_prepare(data, flag='price', oversampling='s', test_size=0.2, seed=1000):
    """
    由于数据集存在不均衡的现象[点击数：未点击数=1:8],所以对训练集数据进行处理，处理方式为对训练集中对点击数据进行随机过采样
    划分训练集和测试集，测试集和训练集分别占比3:7 且按照lable的比例进行抽取
    """
    print('划分测试集和训练集...')
    if oversampling == 's':
        ros = RandomOverSampler(random_state=0)
        new_train_vec, new_train_label = ros.fit_sample(data.loc[:, ~ (data.columns == flag)], data[flag])
        train_x, test_x, train_y, test_y = train_test_split(new_train_vec, new_train_label, test_size=test_size,
                                                            random_state=seed)
    else:
        train_x, test_x, train_y, test_y = train_test_split(data.loc[:, ~ (data.columns == flag)], data[flag],
                                                            test_size=test_size, random_state=seed)
    return train_x, test_x, train_y, test_y



#X_train, X_test, y_train, y_test = data_prepare(data_bj, oversampling='s',test_size=0.2, seed=1000)

        logger.info("#------ 定义线性回归模型-------#")
        model_linear = LinearRegression(fit_intercept=True, normalize=False,copy_X=True, n_jobs=1)
        model_linear = model_linear.fit(X_train,y_train)
        model_linear.predict(X_test)
        model_linear.get_params()
        score = model_linear.score(X_test, y_test)



        lgb.plot_importance(gbm, max_num_features=30, importance_type='split')
        plt.show()
        importance = pd.DataFrame()
        importance['Feature'] = self.X_train.columns.values
        importance['ImportanceWeight'] = gbm.feature_importance(importance_type='split')
        importance['ImportanceGain'] = gbm.feature_importance(importance_type='gain')
        importance.sort_values(by='ImportanceWeight', ascending=False, inplace=True)
        importance.head()


def do_fit(model):
    model.fit(X_train, y_train)
    return model


def do_predict(self, model):
    model.predict(self.X_test)
    # 获得这个模型的参数
    model.get_params()
    # 为模型进行打分
    score = model.score(self.X_test, self.y_test)  # 线性回归：R square； 分类问题： acc
    print(score)
    return score


logger.info("#------ 定义lasso模型-------#")
model_lasso = LassoCV(alphas=np.linspace(0.1, 1000, 10000))
logger.info("#------ fit lasso模型-------#")
model_lasso = self.do_fit(model_lasso)
logger.info("#------ predict lasso模型-------#")
model_lasso_score = self.do_predict(model_lasso)

from sklearn.linear_model import LassoCV
model_lasso = LassoCV(alphas = np.linspace(0.1,1000,10000)).fit(X_train,y_train)
model_lasso.alpha_
coef = pd.Series(model_lasso.coef_, index = X_train.columns)# .coef_ 可以返回经过学习后的所有 feature 的参数。
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])#选头尾各10条，.sort_values() 可以将某一列的值进行排序。
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

#https://blog.csdn.net/previous_moon/article/details/71376726
#1 / (2 * n_samples) * ||y - Xw||^2_2+ alpha * l1_ratio * ||w||_1+ 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
from sklearn.linear_model import ElasticNetCV
#得到拟合模型，其中x_train,y_train为训练集
ENSTest = ElasticNetCV(alphas=np.linspace(0.1,10,100), l1_ratio=np.linspace(0.0,1,101),  max_iter=50000).fit(X_train,y_train)
#利用模型预测，x_test为测试集特征变量
y_prediction = ENSTest.predict(X_test)
r2_score(y_test,y_prediction)

'''
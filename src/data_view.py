import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

#判断分布
data = np.random.randn(100)
stats.probplot(data,dist='norm',plot=pylab)
pylab.show()
# 标签-属性交会图
y = np.random.randint(0,2,100)
plt.scatter(data,y,alpha=0.5,s=120)
plt.show()
#箱线图
data = np.random.randn(100,8)
pylab.boxplot(data)
plt.xlabel("attibute index")
plt.ylabel("quartile ranges")
plt.show()

#彩色平行坐标图
data = pd.DataFrame(data)
summary = data.describe()

for i in range(data.shape[0]):
    datarow = data.iloc[i,:-1]
    labelColor = (data.iloc[i,-1]-min(data.iloc[:,-1]))/(max(data.iloc[:,-1])-min(data.iloc[:,-1]))
    datarow.plot(color=plt.cm.RdYlBu(labelColor),alpha =0.5)
plt.xlabel("attribute index")
plt.ylabel("attibute values")
plt.show()

#数据相关性热力图
corr = data.corr()
plt.pcolor(corr)
plt.show()

import gplearn as gp
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from statistics import mean
from math import sqrt

if __name__ == '__main__':

 rng = check_random_state(0)
 boston = load_boston()
 perm = rng.permutation(boston.target.size)
 boston.data = boston.data[perm]
 boston.target = boston.target[perm]
 function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min','sin','cos']

 gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=None)
 gp.fit(boston.data[:300, :], boston.t


# 离散型变量统计
def descrete_var(data ,title):  #=data_bj['DOM']
    series = pd.Series(data.values)
    pd.isna(series).any()
    series[ pd.isna(series) ]
    missing_ratio = sum(pd.isna(series)) / len(series)
    print(title+"变量   missing ratio: %.4f "%(sum(pd.isna(series)) / len(series)))
    #series  = series.astype('int')
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
    plt.title(title+ "变量  missing ratio: %.4f "%(sum(pd.isna(series)) / len(series)))
    plt.savefig('./pictures/' + title + '.jpg')
    plt.show()
    return missing_ratio







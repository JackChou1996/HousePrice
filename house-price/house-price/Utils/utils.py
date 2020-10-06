import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from scipy.special import boxcox1p


# -----Part1 数据载入----- #
def load_data(path, file_name='', encoding='utf8', is_csv=True):
    file_ = path + file_name
    if is_csv:
        data = pd.read_csv(file_, low_memory=False, encoding=encoding)
    else:
        data = pd.read_excel(file_, encoding=encoding)
    variable = data.columns.tolist()
    print('数据集 %s 加载成功' % file_name)
    print('变量名为:', variable)
    return data


# -----Part2 数据预处理-----#
def nan_data_rate(df, n, ascending_=False, origin=True):
    """
    【Function】缺失率统计函数 nan_data_rate
    :param df: 需要处理的数据框
    :param n: 显示变量个数
    :param ascending_: 按缺失程度上升还是下降表示
    :param origin: 是否显示无缺失值失变量
    :return: 返回前n个缺失变量缺失率
    """
    if n > len(df.columns):  # 判断显示个数是否多于变量总数,如果超过则默认显示全部变量
        print('显示变量个数多于变量总数%i,将显示全部变量' % (len(df.columns)))
        n = len(df.columns)
    na_rate = df.isnull().sum() / len(df) * 100  # 统计各变量缺失率
    if origin:  # 判断为真则显示无缺失值的变量
        na_rate = na_rate.sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    else:  # 判断为负则显示只有缺失值的变量
        na_rate = na_rate.drop(na_rate[na_rate == 0].index).sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    return missing_data.head(n)


# 绘制缺失率直方图
def show_bar(x, y, title):
    sns.set_context('talk')
    fig = plt.figure(figsize=(30, 10))
    ax = sns.barplot(x=x, y=y)
    ax.set_title('Rate of Death Causes')
    ax.set_xticklabels(x, rotation=90)
    plt.show()


# 【Function2】检验字符串是否在变量之中
def reset_str(df, columns='', item_lst=[], new_name=''):
    """
    【Function】检验字符串是否在变量之中
    :param df: 需要处理的数据框
    :param columns: 需要检查的变量名
    :param item_lst: list[str] 需要坚持的字段列表
    :param new_name: 新生成的变量名
    :return: 处理后的数据框
    """
    for item in item_lst:
        i = 0
        while i < len(df[columns]):
            if item in df[columns].iloc[i,]:
                df[new_name].iloc[i,] = item
            i += 1
    return df.copy()


# 【Function3】描述性统计分析计算函数
def describle_data(df):
    """
    【Function】描述性统计分析计算函数
    :param df: 需要描述的数据框
    :return: 返回数值型变量信息,分类型变量信息
    """
    # num_lst:储存数值型数据的均值等信息
    # class_lst:储存分类型数据的类别等信息
    num_lst = []
    class_lst = []
    for col in df.columns:
        if df[col].dtypes != object:
            # 均值水平度量
            mean_ = df[col].mean()
            median_ = df[col].median()
            mode_ = df[col].mode()
            # 集中趋势度量
            std_ = df[col].std()
            iqr_ = df[col].quantile(0.75) - df[col].quantile(0.25)
            range_ = df[col].max() - df[col].min()
            temp = [col, mean_, median_, mode_, std_, iqr_, range_]
            num_lst.append(temp)
        else:
            # 分类变量计数处理
            counts = df[col].value_counts()
            class_lst.append(counts)
    num_detials = pd.DataFrame(num_lst, columns=['变量', '均值', '中位数', '众数', '标准差', '四分位差', '极差'])
    num_detials.index = num_detials['变量']
    num_detials.drop('变量', axis=1, inplace=True)
    return num_detials, class_lst


# 统计各个区的房价中位数
def get_median(df, val, tar):
    """
    :param df: 需要分组统计的数据框
    :param val: 分组变量
    :param tar: 需要统计的变量
    :return: pd.series
    """
    return df.groupby(val)[tar].median()


def get_time_median(data, val, tar):
    time = ['%s' % i for i in range(2000, 2017)]
    print(time)
    dic = {}
    for _ in time:
        # 提取指定年份数据
        temp = data[data[val] == _]
        temp = temp.loc[temp[val] != '亦庄']
        # 调用函数获取分组信息
        res = get_median(temp, val, tar)
        dic[_] = [list(z) for z in zip(res.index, res.values)]
    return dic


# 【Function4】 数值型数据偏态性处理
def stat_skew(df, columns=[], n=10, ascending=False):
    """
    【Function】计算各个变量的偏态系数
    :param df: 需要处理的数据框
    :param columns:  需要处理的变量
    :param n: 需要显示的变量个数
    :param ascending: 显示顺序,默认降序
    :return:
    """
    if not columns:
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
    """
    【Function】修正各个变量的偏态性
    :param df: 需要处理的数据
    :param skewness: 变量偏态性值数据框
    :param threshold: 偏态性认证的阈值
    :param k: boxcox变换中的lambda参数
    :return: 处理后的数据框
    """
    skewness_re = skewness[abs(skewness) > threshold]
    print("There are {} skewed numerical features to Box-Cox transform".format(skewness_re.shape[0]))
    skewed_features = skewness_re.index
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], k)
    return df


def max_min_scale(data):
    return (data - min(data)) / (max(data) - min(data))

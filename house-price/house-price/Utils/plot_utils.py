# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:03:41 2019

@author: DELL
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import matplotlib.cm as cm

from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import brewer
from bokeh.palettes import Spectral10
from bokeh.transform import dodge
from bokeh.layouts import gridplot

# 导入pyecharts绘图工具
from pyecharts import options as opts
from pyecharts.charts import Map, Page, Timeline, Grid
from pyecharts.charts import Geo, Bar, Line, Gauge, Pie
from pyecharts.globals import ChartType, SymbolType
from pyecharts.globals import ThemeType


def kde_plot(background, df, lng, lat, title):
    bg = imageio.imread(background)
    fig, axes = plt.figure(figsize=(15, 15))
    axes.imshow(bg)
    sns.kdeplot(df[lat], df[lng], n_levels=25, cmap=cm.Reds, alpha=0.8)
    plt.title(title)


# 自定义函数绘制直方图和QQ图
def distribute_plot(df, columns=[]):
    # lst:创建列表用于储存变量均值,方差信息
    lst = []
    for col in columns:
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        # 拟合正太分布均值,方差参数
        (mu, sigma) = stats.norm.fit(df[col], fit=stats.norm)
        print('变量{}:$\mu$ = {:.3f} and $\sigma$ = {:.3f}'.format(col, mu, sigma))
        # 绘制直方图
        sns.distplot(df[col], fit=stats.norm)
        plt.legend(['Normal dist.($\mu=$ {:.3f} and $\sigma=$ {:.3f})'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title('%s distribution' % col)
        plt.show()
        # 绘制QQ图
        plt.figure(figsize=(8, 6))
        stats.probplot(df[col], plot=plt)
        lst.append((col, mu, sigma))
        plt.show()
    return lst


# 变量间相关性分析
def corr_plot(df, columns=[]):
    data_cor = df[columns]
    corrmat = data_cor.corr()
    # 设置下三角样式
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # 创建调试盘
    cmap = sns.diverging_palette(0.5, 10, as_cmap=True)
    # 绘制热力图
    # plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, mask=mask, cmap=cmap, annot=True, linewidths=0.2,
                center=100, vmax=1, vmin=0, square=True)
    return


# 自定义创建箱型图的函数
def create_box(df, x_name='', y_name='', title='', hue='', order_=[], scatter_=True):
    sns.boxplot(x=x_name, y=y_name, data=df,
                hue=hue,
                linewidth=2,  # 线宽
                width=0.8,  # 箱子之间的间隔比例
                fliersize=3,  # 异常点大小
                palette='hls',  # 设置调色板
                whis=1.5,  # 设置IQR
                notch=False,  # 设置是否以中值做凹槽
                order=order_)  # 筛选类别
    if scatter_:
        sns.swarmplot(x=x_name, y=y_name, data=df, color='k', size=3, alpha=0.6)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    return


def timeline_map(data) -> Timeline:
    tl = Timeline()
    for price in data:
        geo_ = (
            Geo(init_opts=opts.InitOpts(width=1200, height=800)
                )
                .add_schema(maptype="北京")
                .add(
                "房屋均价",
                data[price],
                type_=ChartType.HEATMAP,
            )
                .add(
                "地理坐标",
                data[price],
                type_=ChartType.EFFECT_SCATTER, blur_size=8, color='yellow',
            )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
                .set_global_opts(title_opts=opts.TitleOpts(title="2000-2017年北京市各区房价(平米)中位数分布图"),
                                 visualmap_opts=opts.VisualMapOpts(),
                                 legend_opts=opts.LegendOpts(pos_right='right'))
        )
        tl.add(geo_, price)
    return tl


# 散点图+边缘分布图
def create_sca_join(df, x='', y=''):
    fig = sns.JointGrid(x=x, y=y, data=df)
    fig.plot_joint(plt.scatter, color='m', edgecolor='white')  # 设置框内图表scatter
    fig.ax_marg_x.hist(df[x], color="b", alpha=0.6)
    fig.ax_marg_y.hist(df[y], color='r', alpha=0.6, orientation="horizontal")
    return ()


# 矩阵散点图
"""
sns.set_style("white")
num_var_lst = ['damageDealt', 'heals', 'walkDistance', 'weaponsAcquired']
pubg_mat = pubg_train[num_var_lst]
sns.pairplot(pubg_mat, kind='scatter', diag_kind='hist', palette='husl', size=4)
"""

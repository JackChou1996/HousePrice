import pandas as pd

# 导入自定义工具
import Utils.utils as utils
import Utils.plot_utils as plot_utils

path = './datasets/house-price/'

# 读取数据集
bj_house = utils.load_data(path, file_name='bj_house.csv')
# sh_house = utils.load_data(path, file_name='shanghai_7w.xlsx', is_csv=False)

# 查看数据集信息
na_data = utils.nan_data_rate(bj_house, 10)
num_lst, class_lst = utils.describle_data(bj_house)

# 绘制缺失率直方图
utils.show_bar(na_data.index, na_data['Missing_Ratio'], "Rate of Missing Value")

# ----- 填补缺失值 ----- #
# 房屋的厨房数，卫生间数。客厅数等都与住房面积高度相关,利用住房面积,对数据集分组用于填补缺失
squ_bins = [-1, 20, 50, 90, 120, 160, 200, 2000]
bj_house['squ_level'] = pd.cut(bj_house['square'], bins=squ_bins,
                               labels=['20平以下', '20-50', '50-90', '90-120', '120-160', '160-200', '200平以上'])
# livingRoom 缺失值填补
bj_house.loc[bj_house['livingRoom'] == '#NAME?', 'livingRoom'] = None
bj_house['livingRoom'] = bj_house.groupby('squ_level')['livingRoom'].transform(lambda x: x.fillna(x.median()))
bj_house['livingRoom'] = bj_house['livingRoom'].map(lambda x: int(x))
# drawingRoom 缺失值填补
bj_house['drawingRoom'] = bj_house.groupby('squ_level')['drawingRoom'].transform(lambda x: x.fillna(x.median()))
bj_house['drawingRoom'] = bj_house['drawingRoom'].map(lambda x: int(x))
# kitchen 缺失值填补
bj_house['kitchen'] = bj_house.groupby('squ_level')['kitchen'].transform(lambda x: x.fillna(x.median()))
bj_house['kitchen'] = bj_house['kitchen'].map(lambda x: int(x))
# bathRoom 缺失值填补
bj_house['bathRoom'] = bj_house.groupby('squ_level')['bathRoom'].transform(lambda x: x.fillna(x.median()))
bj_house['bathRoom'] = bj_house['bathRoom'].map(lambda x: int(x))
# buildingType 缺失值填补
bj_house['buildingType'] = bj_house['buildingType'].fillna('无类型')
# renovationCondition 缺失值填补
bj_house['renovationCondition'] = bj_house['renovationCondition'].fillna('其他')

# ----- 填补缺失值 ----- #
# 异常值处理
# bj_house.loc[bj_house['constructionTime'] == '未知', 'constructionTime'] = None

# ----- 特征工程 ----- #
# 变量扩充 新增 房间总数 和 起居室占比
bj_house['num_room'] = bj_house['livingRoom'].map(lambda x: int(x)) + bj_house['drawingRoom'].map(lambda x: int(x)) + \
                       bj_house['kitchen'].map(lambda x: int(x)) + bj_house['bathRoom'].map(lambda x: int(x))
bj_house['living_rate'] = bj_house['livingRoom'].map(lambda x: int(x)) / bj_house['num_room']

# 变量替换
# buildingType 变量替换
bj_house.loc[bj_house['buildingType'] == 1, 'buildingType'] = '塔楼'
bj_house.loc[bj_house['buildingType'] == 2, 'buildingType'] = '平房'
bj_house.loc[bj_house['buildingType'] == 3, 'buildingType'] = '板塔结合'
bj_house.loc[bj_house['buildingType'] == 4, 'buildingType'] = '板楼'

# renovationCondition 变量替换
bj_house.loc[bj_house['renovationCondition'] == 1, 'renovationCondition'] = '其他'
bj_house.loc[bj_house['renovationCondition'] == 2, 'renovationCondition'] = '毛坯'
bj_house.loc[bj_house['renovationCondition'] == 3, 'renovationCondition'] = '简装'
bj_house.loc[bj_house['renovationCondition'] == 4, 'renovationCondition'] = '精装'

# district 变量替换
bj_house.loc[bj_house['district'] == 1, 'district'] = '东城'
bj_house.loc[bj_house['district'] == 2, 'district'] = '丰台'
bj_house.loc[bj_house['district'] == 3, 'district'] = '亦庄'
bj_house.loc[bj_house['district'] == 4, 'district'] = '大兴'
bj_house.loc[bj_house['district'] == 5, 'district'] = '房山'
bj_house.loc[bj_house['district'] == 6, 'district'] = '昌平'
bj_house.loc[bj_house['district'] == 7, 'district'] = '朝阳'
bj_house.loc[bj_house['district'] == 8, 'district'] = '海淀'
bj_house.loc[bj_house['district'] == 9, 'district'] = '石景山'
bj_house.loc[bj_house['district'] == 10, 'district'] = '西城'
bj_house.loc[bj_house['district'] == 11, 'district'] = '通州'
bj_house.loc[bj_house['district'] == 12, 'district'] = '门头沟'
bj_house.loc[bj_house['district'] == 13, 'district'] = '顺义'

# buildingStructure 变量替换
bj_house.loc[bj_house['buildingStructure'] == 1, 'buildingStructure'] = '未知结构'
bj_house.loc[bj_house['buildingStructure'] == 2, 'buildingStructure'] = '混和结构'
bj_house.loc[bj_house['buildingStructure'] == 3, 'buildingStructure'] = '砖木结构'
bj_house.loc[bj_house['buildingStructure'] == 4, 'buildingStructure'] = '砖混结构'
bj_house.loc[bj_house['buildingStructure'] == 5, 'buildingStructure'] = '钢结构'
bj_house.loc[bj_house['buildingStructure'] == 6, 'buildingStructure'] = '钢混结构'

# 绘制缺失率直方图
utils.show_bar(na_data.index, na_data['Missing_Ratio'], "Rate of Missing Value")

# 只研究2000年-2016年的房价信息
time_bins = ['1999', '2004', '2008', '2012', '2017']
bj_house_period = pd.cut(bj_house['constructionTime'], bins=time_bins,
                         labels=['2000-2004', '2005-2008', '2009-2012', '2013-2016'])
bj_house_period.value_counts()

"""
# 统计不同行政区不同年份房价中位数
time_price = utils.get_time_median(bj_house, 'district', 'price')
# 绘制不同行政区域不同年份房价中位数热力图
plot_utils.timeline_map(time_price).render('beijing_price_map')
"""

# 绘制各年份有无地铁平均房价分布图
plot_utils.create_box(bj_house, 'period', 'price', 'The Boxplot of House Price from 2000 to 2016', 'subway',
                      order_=['2000-2004', '2005-2008', '2009-2012', '2013-2016'], scatter_=False)
# 绘制建筑结构和房屋每平米价格分布图
plot_utils.create_box(bj_house, 'buildingStructure', 'price',
                      'The Boxplot of House Price in deffierent period and building structure', 'period',
                      order_=[1, 2, 3, 4, 5, 6], scatter_=False)
# 绘制建筑类型和房屋每平米价格分布图
plot_utils.create_box(bj_house, 'buildingType', 'price',
                      'The Boxplot of House Price in deffierent period and building type', 'period',
                      order_=[1, 2, 3, 4, 5, 6], scatter_=False)

plot_utils.distribute_plot(df=bj_house, columns=['price', 'square', 'communityAverage', 'ladderRatio'])

plot_utils.kde_polt(background='./datasets/beijing_map.png', df=bj_house, lng='Lng', lat='Lat',
                    title='the distribution of Beijing house')

# 判断成交总价是否等于每平米价格x建筑面积 不等于
sum(bj_house['totalPrice'] - bj_house['square'] * bj_house['price'])
# 但有极强的共线性，因此用这个指标进行预测是无意义的
bj_house['avg_price'] = bj_house['totalPrice'] / bj_house['square']
plot_utils.create_sca_join(bj_house, 'avg_price', 'price')

bj_house['avg_community'] = utils.max_min_scale(bj_house['communityAverage'])
plot_utils.create_sca_join(bj_house, 'avg_community', 'sta_price')

from datetime import datetime, date, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import time
import pickle
import logging
logging.basicConfig(format="%(asctime)15s [%(levelname)s]: %(message)s", level=logging.DEBUG)
logger = logging.getLogger()

DATA_STORAGE_PATH_SH = 'data_before_map_sh.pkl'
DATA_STORAGE_PATH_BJ = 'data_before_map_bj.pkl'

EXTENDED_COLUMNS = ['livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor','floor_num',
                      'square', 'unitStructure', 'innerArea', 'buildingType', 'orient',
                      'buildingStructure', 'renovationCondition', 'ladderRatio', 'elevator', 'ownYear']
EXTENDED_COLUMNS1 = ['onBoard year','onBoard month','trading_property','to_lastTrade',
                     'use_property','fiveYearsProperty','owner','mortgage','update_certificate']



def basic(s):

    def room(s) -> list:
        try:
            room = s
            l = re.findall(r'\d+', room) if (re.findall(r'\d+', room)) else [np.nan] * 4
            l = list(map(int, l))
            return l
        except:
            print("room",s); time.sleep(1)
            return ([np.nan] * 4)

    def floor(s):
        try:
            sub_str = s  # s = '所在楼层：中层 (共17层)'  #l = ['高楼层','底层','中楼层','低楼层','顶层']    l = str("".join(l))
            l = (re.search('：(.*?层)', sub_str).group(1)) if (re.search('：(.*?层)', sub_str).group(1)) else np.nan
            num = int(re.search('\d+', sub_str).group(0))
            return pd.Series([l,num],index = ['floor','floor_num'])
        except:
            print("floor", s); time.sleep(1)
            return pd.Series([np.nan,np.nan],index = ['floor','floor_num'])

    def square(s):
        try:  # square
            sub_str = s
            l = re.search('\d+.*\d+', sub_str).group()
            return (float(l))
        except:
            print("area", s);time.sleep(1)
            return np.nan

    def unitStructure(s):
        try:  # unitStructure
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            print("unitStructure", s);time.sleep(1)
            return np.nan

    def innerArea(s):
        try:  # 'innerArea'
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            print("innerArea", s);time.sleep(1)
            return np.nan

    def buildingType(s):
        try:  # 'buildingType'
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            print("buildingType", s);time.sleep(1)
            return np.nan

    def orient(s):
        try:  # 'orient',
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            print("orient", s);time.sleep(1)
            return np.nan

    def buildingStruction(s):
        try:  # 'buildingStructure  buildingStruction'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            print("buildingStructure", s);time.sleep(1)
            return np.nan

    def renovationCondition(s):
        try:  # 'renovationCondition'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            print("renovationCondition", s);time.sleep(1)
            return np.nan

    def ladderRatio(s):
        try:  # 'ladderRatio'
            dic = {'一': 1, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12,
                   '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
                   '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28,
                   '二十九': 29, '三十': 30, '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35, '三十六': 36,
                   '三十七': 37, '三十八': 38, '三十九': 39, '四十': 40, '四十一': 41, '四十二': 42, '四十三': 43, '四十四': 44,
                   '四十五': 45, '四十六': 46, '四十七': 47, '四十八': 48, '四十九': 49, '五十': 50, '五十一': 51, '五十二': 52,
                   }
            sub_str = s
            ladder = dic[sub_str[5]]
            num = dic[sub_str[7:-1]]
            return (num / ladder)
        except:
            # print("ladderRatio", s);time.sleep(1)
            return np.nan

    def elevator(s):
        try:  # 'elevator'
            sub_str = s
            l = sub_str[-1]
            return l
        except:
            print("elevator", l);time.sleep(1)
            return np.nan

    def ownYear(s):
        try:  # 'ownYear'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            print("ownYear", l); time.sleep(1)
            return np.nan

    res = pd.Series(index=EXTENDED_COLUMNS, data=[np.nan] * len(EXTENDED_COLUMNS))

    if pd.isna(s):
        return res

    #lst = list(filter(lambda x: x.strip(),  s.split("/")))
    lst = list(map(lambda x: x.strip(), s.split("/")))
    for item in lst:
        try:
            if (item[:4] == "房屋户型"):
                # 这里也要修改一下
                res.loc[['livingRoom', 'drawingRoom', 'kitchen', 'bathRoom']] = room(item)
            elif (item[:4] == "所在楼层"):
                res.loc[['floor','floor_num']] = floor(item)
            elif (item[:4] == "建筑面积"):
                res.loc['square'] = square(item)
            elif (item[:4] == "户型结构"):
                res.loc['unitStructure'] = unitStructure(item)
            elif (item[:4] == "套内面积"):
                res.loc['innerArea'] = innerArea(item)
            elif (item[:4] == "建筑类型"):
                res.loc['buildingType'] = buildingType(item)
            elif (item[:4] == "房屋朝向"):
                res.loc['orient'] = orient(item)
            elif (item[:4] == "建筑结构"):
                res.loc['buildingStructure'] = buildingStruction(item)
            elif (item[:4] == "装修情况"):
                res.loc['renovationCondition'] = renovationCondition(item)
            elif (item[:4] == "梯户比例"):
                res.loc['ladderRatio'] = ladderRatio(item)
            elif (item[:4] == "配备电梯"):
                res.loc['elevator'] = elevator(item)
            elif (item[:4] == "产权年限"):
                res.loc['ownYear'] = ownYear(item)
            else:
                pass
        except Exception as e:
            raise e
    return res

def trading(s):

    def onBoard(s):
        try:
            dt = pd.to_datetime(s[5:15])
            return [dt.year,dt.month]
        except:
            print("onBoard", s);time.sleep(1)
            return [np.nan]*2
    def trading_property(s):
        try:
            return  s[5:]
        except:
            print("trading_property", s);time.sleep(1)
            return np.nan
        #['挂牌时间：2018-03-15', '交易权属：商品房', '上次交易：暂无数据', '房屋用途：普通住宅', '房屋年限：暂无数据', '产权所属：非共有', '抵押信息                                                              无抵押', '房本备件：未上传房本照片']

    def to_lastTrade(board,last_trade):   #last_trade='上次交易：2018-03-07'   board= '挂牌时间：2017-12-13'
        try:
            dt_board = pd.to_datetime(board[5:15])
            dt_lastTrade = pd.to_datetime(last_trade[5:15])
            return (dt_board-dt_lastTrade).days
        except:
            print("to_lastTrade", board,last_trade); #time.sleep(1)
            return np.nan
    def use_property(s):
        try:
            return s[5:]
        except:
            print("use_property", s);time.sleep(1)
            return np.nan
    def fiveYearsProperty(s):  #'房屋年限：未满两年'
        try:
            return s[5:]
        except:
            print('fiveYearsProperty', s);time.sleep(1)
            return np.nan
    def owner(s):  #'产权所属：非共有',
        try:
            return s[5:]
        except:
            print('owner', s);time.sleep(1)
            return np.nan
    def mortgage(s):  #s = '抵押信息                                                              无抵押'
        try:
            return s.split(' ')[-1]
        except:
            print('mortgage', s);time.sleep(1)
            return np.nan
    def update_certificate(s):  #s = '房本备件：已上传房本照片'
        try:
            return s[5:]
        except:
            print('update_certificate', s);time.sleep(1)
            return np.nan

    res = pd.Series(index=EXTENDED_COLUMNS1, data=[np.nan] * len(EXTENDED_COLUMNS1))

    if pd.isna(s):
        return res
    lst = list(map(lambda x: x.strip(), s.split('/')))
    #lst = list(filter(lambda x: x.strip(),  s.split('/')))
    board_time = ''
    for item in lst:
        try:
            #'onBoard year','onBoard month','trading_property','to_lastTrade','use_property','fiveYearsProperty','owner','mortgage','update_certificate'
            if (item[:4] == "挂牌时间"):
                # 这里也要修改一下
                board_time = item
                res.loc[['onBoard year','onBoard month']] = onBoard(item)
            elif (item[:4] == "交易权属"):
                res.loc['trading_property'] = trading_property(item)
            elif (item[:4] == "上次交易" and len(board_time)>0):
                #print(lst)
                res.loc['to_lastTrade'] = to_lastTrade(board_time,item)
            elif (item[:4] == "房屋用途"):
                res.loc['use_property'] = use_property(item)
            elif (item[:4] == "房屋年限"):
                res.loc['fiveYearsProperty'] = fiveYearsProperty(item)
            elif (item[:4] == "产权所属"):
                res.loc['owner'] = owner(item)
            elif (item[:4] == "抵押信息"):
                res.loc['mortgage'] = mortgage(item)
            elif (item[:4] == "房本备件"):
                res.loc['update_certificate'] = update_certificate(item)
            else:
                pass
        except Exception as e:
            raise e
    return res


def init_data(dump=False):
    def first_nums(s):
        try:
            l = re.search('\d+', s).group()
            return int(l)
        except:
            return np.nan

    def geo(s):
        try:
            l = s.split(',')
            l = list(map(float, l))
            return pd.Series(dict(zip(['Lng', 'Lat'], l)), index=['Lng', 'Lat'])
        except:
            return pd.Series(dict(zip(['Lng', 'Lat'], [np.nan, np.nan])), index=['Lng', 'Lat'])
    logger.info("Reading Shanghai Excel...")
    data = pd.read_excel("Shanghai(7w).xlsx", encoding='gbk')
    logger.info("Reading Excel finished.")
    dic = {'编号':'id', '单价':'price','总价':'totalPrice','首付':'first_installment','基本属性':'basic_property','建成时间':'constructionTime','交易属性':'tradings','经纬度':'loglat','行政区':'district','板块':'field','小区名称':'cummunity_name'}
    data.columns = list(dic.values())
    data['first_installment'] = data['first_installment'].apply(first_nums, )
    data['constructionTime'] = data['constructionTime'].apply(first_nums, )
    #  sorted( data['constructionTime'].unique()) 1911-2016
    # l = ( pd.Series(data['constructionTime'].unique()).dropna() ); l = list(l); l = l.sort()
    # data['buildingType'] = data.建成时间.apply(type_of_building,)
    # {nan, '塔楼', '暂无数据', '板塔结合', '板楼'}  set(data.建成时间.apply(type_of_building,))
    data = data.reindex(columns=list(data.columns) + ['Lng', 'Lat'])
    data[['Lng', 'Lat']] = data['loglat'].apply(geo, )
    data = data.drop(['loglat'],axis=1)

    orient_columns = list(data.columns)
    orient_columns.extend(EXTENDED_COLUMNS)
    data = data.reindex(columns=orient_columns)

    logger.info("Mapping...")
    data[EXTENDED_COLUMNS] = data['basic_property'].apply(basic, )
    data[EXTENDED_COLUMNS1] = data['tradings'].apply(trading, )
    data = data.drop(['basic_property', 'tradings'],axis=1)
    #
    logger.info("Mapping finished.")

    if dump:
        pickle.dump(data, open(DATA_STORAGE_PATH_SH, 'wb'))

    return data

def init_data_bj(dump = False):
    def floor_num(s):
        try:
            return int(s.split(' ')[1])
        except:
            return np.nan
    def floor(s):
        try:
            return  s.split(' ')[0]
        except:
            return np.nan

    logger.info("Reading Beijing Excel...")
    data_bj = pd.read_csv('Beijing(31w).csv')
    logger.info("Reading Excel finished.")
    data_bj = data_bj.drop(['url'],axis = 1)
    data_bj['floor_num'] =  data_bj['floor'].apply(lambda x: (x.split(' ')[1]) if not pd.isna(x) and len( x.split(' '))>1 and not pd.isna(x.split(' ')[1] ) else np.nan)
    data_bj['floor'] =  data_bj['floor'].apply(floor)
    if dump:
        pickle.dump(data_bj, open(DATA_STORAGE_PATH_BJ, 'wb'))
    return data_bj



def load_data_from_disk(DATA_STORAGE_PATH):
    assert os.path.exists(DATA_STORAGE_PATH), '不存在路径{}'.format(DATA_STORAGE_PATH)
    logger.info("loading data ...")
    data = pickle.load(open(DATA_STORAGE_PATH, 'rb'))
    logger.info("loading data finished.")
    return data

if __name__ == "main":
    DATA_STORAGE_PATH_SH = './data/data_before_map_sh.pkl'
    DATA_STORAGE_PATH_BJ = './data/data_before_map_bj.pkl'

    HOT_LOAD = True
    DUMP_DATA = True

    if not HOT_LOAD or not os.path.exists(DATA_STORAGE_PATH_SH):
        data_sh = init_data(dump=DUMP_DATA)
    else:
        data_sh = load_data_from_disk(DATA_STORAGE_PATH_SH)

    if not HOT_LOAD or not os.path.exists(DATA_STORAGE_PATH_BJ):
        data_bj = init_data_bj(dump=DUMP_DATA)
    else:
        data_bj = load_data_from_disk(DATA_STORAGE_PATH_BJ)



    columns_sh = data_sh.columns
    '''
    ['id', 'price', 'totalPrice', 'first_installment', 
           'constructionTime', 'trading_property', 'adminDistrict', 'field',
           'cummunity_name', 'Lng', 'Lat', 'livingRoom', 'drawingRoom', 'kitchen',
           'bathRoom', 'floor', 'square', 'unitStructure', 'innerArea',
           'buildingType', 'orient', 'buildingStruction', 'renovationCondition',
           'ladderRatio', 'elevator', 'ownYear', 'onBoard year', 'onBoard month',
           'to_lastTrade', 'use_property', 'fiveYearsProperty', 'owner',
           'mortgage', 'update_certificate']
    '''
    columns_bj = data_bj.columns
    '''
    ['url', 'id', 'Lng', 'Lat', 'Cid', 'tradeTime', 'DOM', 'followers',
           'totalPrice', 'price', 'square', 'livingRoom', 'drawingRoom', 'kitchen',
           'bathRoom', 'floor', 'buildingType', 'constructionTime',
           'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
           'fiveYearsProperty', 'subway', 'district', 'communityAverage',
           'floor_num']
    {'Cid', 'DOM', 'Lat', 'Lng', 'bathRoom', 'buildingStructure', 'buildingType',
     'communityAverage', 'constructionTime', 'district', 'drawingRoom', 'elevator',
     'fiveYearsProperty', 'floor', 'floor_num', 'followers', 'id', 'kitchen', 'ladderRatio',
     'livingRoom', 'price', 'renovationCondition', 'square', 'subway', 'totalPrice',
     'tradeTime', 'url'}
    '''
    common_columns = set(columns_sh) & set(columns_bj)
    '''
    {'Lat', 'Lng', 'buildingType', 'constructionTime', 'livingRoom','drawingRoom',  'kitchen',
    'elevator', 'fiveYearsProperty', 'floor', 'id', 'ladderRatio', 
     'price', 'renovationCondition', 'square', 'totalPrice'}
    '''










"""
---
for test
---
# temp = data.head(10)
# temp.drop(columns=EXTENDED_COLUMNS, inplace=True)
"""



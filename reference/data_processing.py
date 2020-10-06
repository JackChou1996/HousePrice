from datetime import datetime, date, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import time

def first_nums(s):
    try:
        l = re.search('\d+',s).group()
        return int(l)
    except:
        return np.nan

def geo(s):
    try:
        l = s.split(',')
        l = list(map(float, l))
        return pd.Series(dict(zip(['lng','lat'],l  )) ,index=['lng','lat'])
    except:
        return  pd.Series( dict(zip(['lng','lat'],[np.nan,np.nan]  )) ,index=['lng','lat'])


def basic(s):
    #global i
    #print(i)
    #i+=1
    #print(s)
    res = {}
    extend_columns = ['livingRoom', 'drawingRoom', 'kitchen', 'bathroom', 'floor',
                      'square', 'unitStructure', 'innerArea', 'buildingType', 'orient',
                      'buildingStruction', 'renovationCondition', 'ladderRatio', 'elevator', 'ownYear']
    if(pd.isna(s)):
        return pd.Series(dict( zip(extend_columns,[np.nan]*len(extend_columns))),index=extend_columns)
    lst = s.split("/")
    lst = list(map(lambda x:x.strip(),lst))
    lst.remove('')

    def room(s)->list:
        try:
            room = s
            l = re.findall(r'\d+', room) if( re.findall(r'\d+', room) ) else [np.nan]*4
            l = list(map(int,l))
            return l
        except:
            #print("room",s); time.sleep(1)
            return ( [np.nan]*4)
    def floor(s):
        try:
            sub_str = s  #f = '所在楼层：中层 (共17层)'  #l = ['高楼层','底层','中楼层','低楼层','顶层']    l = str("".join(l))
            l = (re.search('：(.*?层)', sub_str).group(1)) if (re.search('：(.*?层)', sub_str).group(1)) else np.nan
            return l
        except:
            #print("floor", s); time.sleep(1)
            return np.nan
    def square(s):
        try: #square
            sub_str = s
            l = re.search('\d+.*\d+', sub_str).group()
            return (float(l))
        except:
            #print("area", s);time.sleep(1)
            return np.nan
    def unitStructure(s):
        try: #unitStructure
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            #print("unitStructure", s);time.sleep(1)
            return np.nan
    def innerArea(s):
        try:  #'innerArea'
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            #print("innerArea", s);time.sleep(1)
            return np.nan
    def buildingType(s):
        try:  #'buildingType'
            sub_str = s
            l = sub_str[5:7]
            return l
        except:
            #print("buildingType", s);time.sleep(1)
            return np.nan
    def orient(s):
        try:  #'orient',
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            #print("orient", s);time.sleep(1)
            return np.nan
    def buildingStruction(s):
        try:  # 'buildingStruction'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            #print("buildingStruction", s);time.sleep(1)
            return np.nan
    def renovationCondition(s):
        try:  #'renovationCondition'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            #print("renovationCondition", s);time.sleep(1)
            return np.nan
    def ladderRatio(s):
        try:  #'ladderRatio'
            dic = {'一': 1, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,'十一':11,'十二':12,
                   '十三':13,'十四':14,'十五':15,'十六':16,'十七':17,'十八':18,'十九':19,'二十':20,
                   '二十一':21,'二十二':22,'二十三':23,'二十四':24,'二十五':25,'二十六':26,'二十七':27,'二十八':28,
                    '二十九':29,'三十':30,'三十一':31,'三十二':32,'三十三':33,'三十四':34,'三十五':35,'三十六':36,
                   '三十七':37,'三十八':38,'三十九':39,'四十':40,'四十一':41,'四十二':42,'四十三':43,'四十四':44,
                   '四十五':45,'四十六':46,'四十七':47,'四十八':48,'四十九':49,'五十':50,'五十一':51,'五十二':52,
                    }
            sub_str = s
            ladder = dic[sub_str[5]]
            num = dic[sub_str[7:-1]]
            return (num / ladder)
        except:
            #print("ladderRatio", s);time.sleep(1)
            return np.nan
    def elevator(s):
        try:  # 'elevator'
            sub_str = s
            l = sub_str[-1]
            return l
        except:
            #print("elevator", l);time.sleep(1)
            return np.nan
    def ownYear(s):
        try:  # 'ownYear'
            sub_str = s
            l = sub_str[5:]
            return l
        except:
            #print("ownYear", l); time.sleep(1)
            return np.nan

    for i in range(len(lst)):
        if(lst[i][:4] =="房屋户型"):
            res.update( dict(zip( ['livingRoom', 'drawingRoom', 'kitchen', 'bathroom'],room(lst[i]) )) )
        elif(lst[i][:4] =="所在楼层"):
            res['floor']= floor(lst[i])
        elif(lst[i][:4] =="建筑面积"):
            res['square'] = square(lst[i])
        elif (lst[i][:4] == "户型结构"):
            res['unitStructure'] = unitStructure(lst[i])
        elif (lst[i][:4] == "套内面积"):
            res['innerArea'] = innerArea(lst[i])
        elif (lst[i][:4] == "建筑类型"):
            res['buildingType'] = buildingType(lst[i])
        elif (lst[i][:4] == "房屋朝向"):
            res['orient'] = orient(lst[i])
        elif (lst[i][:4] == "建筑结构"):
            res['buildingStruction'] = buildingStruction(lst[i])
        elif (lst[i][:4] == "装修情况"):
            res['renovationCondition'] = renovationCondition(lst[i])
        elif (lst[i][:4] == "梯户比例"):
            res['ladderRatio'] = ladderRatio(lst[i])
        elif (lst[i][:4] == "配备电梯"):
            res['elevator'] = elevator(lst[i])
        elif (lst[i][:4] == "产权年限"):
            res['ownYear'] = ownYear(lst[i])
        else:
            pass
    for key in (set(extend_columns)-set(res.keys())):
        res[key] = np.nan
    temp_data = pd.Series(res,index=extend_columns)
    #temp_data = pd.Series([np.nan]*4,index=['livingRoom','drawingRoom','kitchen','bathroom'])
    return temp_data



data  = pd.read_excel("Shanghai(7w).xlsx",encoding='gbk')
data['first_installment'] = data.首付.apply(first_nums,)
data['constructionTime'] = data.建成时间.apply(first_nums,)
#  sorted( data['constructionTime'].unique()) 1911-2016
# l = ( pd.Series(data['constructionTime'].unique()).dropna() ); l = list(l); l = l.sort()
#data['buildingType'] = data.建成时间.apply(type_of_building,)
#{nan, '塔楼', '暂无数据', '板塔结合', '板楼'}  set(data.建成时间.apply(type_of_building,))
data = data.reindex(columns = list(data.columns)+['lng','lat'])
data[['lng','lat']] = data['经纬度'].apply(geo,)


orient_columns= list(data.columns);
extend_columns = ['livingRoom','drawingRoom','kitchen','bathroom','floor',
     'square','unitStructure','innerArea','buildingType','orient',
     'buildingStruction','renovationCondition','ladderRatio','elevator','ownYear']
orient_columns.extend(extend_columns)
data = data.reindex(columns = orient_columns)
data[extend_columns] = data.基本属性.apply(basic,)
#temp = data.基本属性.apply(basic,)

#data[extend_columns] = data.交易属性.apply(basic,)
s = '房屋户型：2室2厅1厨1卫/                                            所在楼层：中楼层 (共6层)/                                            建筑面积：76.58㎡/                                            户型结构：暂无数据/                                            套内面积：暂无数据/                                            建筑类型：板楼/                                            房屋朝向：南/                                            建筑结构：钢混结构/                                            装修情况：其他/                                            梯户比例：一梯两户/                                            配备电梯：无/                                            产权年限：未知/'
for i in range(len(data)):
    s = data.基本属性.iloc[i]
    print(basic(s))
    len(basic(s))
    if(s ==s1):
        break

'''
livingRoom:卧室(间)
drawingRoom:客厅(间)
kitchen:厨房(间)
bathroom:卫生间(间)
totalPrice: 成交价格总额
price：价格
square:建筑面积
unitStructure：户型结构
floor:楼层
buildingType: 建筑类型
buildingStruction: 建筑结构
renovationCondition: 装修情况
ladderRatio: 梯户比例
elevator: 电梯
fiveYearsProperty: 房屋年限
Subway: 1000米内是否有地铁站
District: 行政区
CommunityAverage: 小区均价
innerArea:套内面积
'''




'''
def rooms(s):
    print(s)
    try:
        room = s.split("/")[0]
        l = re.findall(r'\d+', room)
        #temp_data = pd.DataFrame({'livingRoom':l[0], 'drawingRoom':l[1],'kitchen':l[2],'bathroom':[3]})
        temp_data = pd.Series(l,index =['livingRoom','drawingRoom','kitchen','bathroom'] )
    except:
        dict_data ={'livingRoom': np.nan, 'drawingRoom': np.nan, 'kitchen': np.nan, 'bathroom': np.nan}
        #temp_data = pd.DataFrame.from_dict(dict_data, orient='index')
        #temp_data = pd.DataFrame({'livingRoom': np.nan, 'drawingRoom': np.nan, 'kitchen': np.nan, 'bathroom': np.nan})
        temp_data = pd.Series([np.nan]*4,index=['livingRoom','drawingRoom','kitchen','bathroom'])
    return temp_data

def basic(s):
    #print(s)
    res = []
#    l = s.split("/")
#    l = list(map(lambda x:x.strip(),l))
#    l.remove('')
    try:
        room = s.split("/")[0]
        l = re.findall(r'\d+', room) if( re.findall(r'\d+', room) ) else [np.nan]*4
        l = list(map(int,l))
        res.extend(l)
    except:
        print("room",l); time.sleep(1)
        res.extend( [np.nan]*4)
        return -1;
    try:
        sub_str = s.split("/")[1].strip()  #f = '所在楼层：中层 (共17层)'  #l = ['高楼层','底层','中楼层','低楼层','顶层']    l = str("".join(l))
        l = (re.search('：(.*?层)', sub_str).group(1)) if (re.search('：(.*?层)', sub_str).group(1)) else np.nan
        res.append(l)
    except:
        print("floor", l); time.sleep(1)
        res.extend([np.nan])
        return -1;
    try: #square
        sub_str = s.split("/")[2].strip()
        l = re.search('\d+.*\d+', sub_str).group()
        res.append(float(l))
    except:
        print(s)
        print("area", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try: #unitStructure
        sub_str = s.split("/")[3].strip()
        l = sub_str[5:7]
        res.append(l)
    except:
        print("unitStructure", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  #'innerArea'
        sub_str = s.split("/")[4].strip()
        l = sub_str[5:7]
        res.append(l)
    except:
        print("innerArea", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  #'buildingType','orient',
        sub_str = s.split("/")[5].strip()
        l = sub_str[5:7]
        res.append(l)
    except:
        print("buildingType", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  #'orient',
        sub_str = s.split("/")[6].strip()
        l = sub_str[5:]
        res.append(l)
    except:
        print("orient", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  # 'buildingStruction'
        sub_str = s.split("/")[7].strip()
        l = sub_str[5:]
        res.append(l)
    except:
        print("buildingStruction", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  #'renovationCondition'
        sub_str = s.split("/")[8].strip()
        l = sub_str[5:]
        res.append(l)
    except:
        print("renovationCondition", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  #'ladderRatio'
        dic = {'一': 1, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        sub_str = s.split("/")[9].strip()
        ladder = dic[sub_str[5]]
        num = dic[sub_str[7]]
        res.append(num / ladder)
    except:
        print("ladderRatio", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  # 'elevator'
        sub_str = s.split("/")[10].strip()
        l = sub_str[-1]
        res.append(l)
    except:
        print("elevator", l);time.sleep(1)
        res.extend([np.nan])
        return -1;
    try:  # 'ownYear'
        sub_str = s.split("/")[11].strip()
        l = sub_str[5:]
        res.append(l)
    except:
        print("ownYear", l);
        time.sleep(1)
        res.extend([np.nan])
        return -1;
    temp_data = pd.Series(res, index=extend_columns)
    #temp_data = pd.Series([np.nan]*4,index=['livingRoom','drawingRoom','kitchen','bathroom'])
    return 0

'''
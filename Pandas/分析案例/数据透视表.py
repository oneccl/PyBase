"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/10/5
Time: 15:10
Description:
"""

# Python数据透视表
'''
数据透视表（Pivot Table）是一种交互式的表，可以进行某些计算，如求和与计数等。所进行的计算与数据跟数据透视表中的排列有关
之所以称为数据透视表，是因为可以动态地改变它们的版面布置，以便按照不同方式分析数据，也可以重新安排行号、列标和页字段。每一次改变版面布置时，数据透视表会立即按照新的布置重新计算数据。另外，如果原始数据发生更改，则可以更新数据透视表
'''
import numpy as np
import pandas as pd

# 读取数据
df = pd.read_excel(r'C:\Users\cc\Desktop\pivot_table.xlsx')
print(df.to_string())
'''
   商品ID   商品名称   城市    类别    销售量    销售额
0  P022    普洱茶     上海    茶叶     80      32000
1  P002    太阳镜     上海    服装饰品  80      16000
2  P002    太阳镜     北京    服装饰品  90      3400
3  P005    手提包     北京    皮具     30      3000
4  P013    手提包     北京    皮具     30      3000
5  P014    商务皮包   上海    皮具     50      25000
6  P011    面膜      广州    化妆品    120     6000
7  P020    拖鞋      深圳    服装饰品  200     8000
8  P004    电动牙刷   深圳    电器     20      2000
9  P021    零食大礼包  北京    食品     10     1000
'''

# 语法结构：
# pd.pivot_table() 或 df.pivot_table()
'''
pivot_table(data, values, index, columns, aggfunc='mean', fill_value, margins=False, dropna=True, margins_name='All')
- data：DataFrame对象
- values：需要计算的数据列
- index：行层次的分组键，从而形成多级索引，str或list类型
- columns：列层次的分组键，从而形成多级列名，str或list类型
- aggfunc：对数据执行的聚合操作函数，默认mean计算平均（支持应用多个函数）
- fill_value：设定缺失替换值
- margins：是否添加行列的汇总
- dropna：如果列的所有值都为NaN，将不作为计算列，默认True
- margins_name：汇总行列的名称，默认为All
'''

# 1、基本使用

# 1）通过商品ID和商品名称作为索引行，计算每个商品的销售额和销售量（默认计算均值）
data1 = df.pivot_table(index=['商品ID', '商品名称'], values=['销售量', '销售额']).reset_index()
print(data1.to_string())
'''
   商品ID   商品名称   销售量    销售额
0  P002    太阳镜    85.0     9700.0
1  P004   电动牙刷    20.0     2000.0
2  P005    手提包    30.0     3000.0
3  P011     面膜     120.0    6000.0
4  P013    手提包    30.0     3000.0
5  P014   商务皮包    50.0    25000.0
6  P020     拖鞋     200.0    8000.0
7  P021  零食大礼包   10.0     1000.0
8  P022    普洱茶    80.0     32000.0
'''
# 2）通过商品ID和商品名称作为索引行，计算每个商品的销售额和销售量（总和及均值）
# 多个字段一次应用多个函数，使用列表传递
data2 = df.pivot_table(index=['商品ID', '商品名称'], values=['销售量', '销售额'], aggfunc=[sum, np.mean]).reset_index()
print(data2.to_string())
'''
   商品ID   商品名称  sum             mean         
                   销售量    销售额    销售量    销售额
0  P002    太阳镜     170   19400     85.0    9700.0
1  P004   电动牙刷      20   2000     20.0    2000.0
2  P005    手提包      30   3000     30.0    3000.0
3  P011     面膜     120    6000     120.0   6000.0
4  P013    手提包      30   3000     30.0    3000.0
5  P014   商务皮包      50  25000     50.0   25000.0
6  P020     拖鞋     200    8000     200.0   8000.0
7  P021  零食大礼包      10   1000    10.0    1000.0
8  P022    普洱茶      80    32000    80.0   32000.0
'''
# 3）通过商品ID和商品名称作为索引行，计算每个商品的平均销售额和总销售量
# 每个字段应用不同函数，使用字典传递；每个字段一次应用多个函数，使用列表传递
data3 = df.pivot_table(index=['商品ID', '商品名称'], values=['销售量', '销售额'], aggfunc={'销售量': sum, '销售额': np.mean}).reset_index()
print(data3.to_string())
'''
   商品ID   商品名称  销售量    销售额
0  P002    太阳镜    170      9700.0
1  P004   电动牙刷    20      2000.0
2  P005    手提包    30       3000.0
3  P011     面膜     120      6000.0
4  P013    手提包    30       3000.0
5  P014   商务皮包    50      25000.0
6  P020     拖鞋     200      8000.0
7  P021  零食大礼包   10       1000.0
8  P022    普洱茶     80      32000.0
'''

# 2、进阶使用

# 4）汇总排序
# 计算每个城市各类别商品的总销量，并根据类别汇总排序
data4 = df.pivot_table(index='类别', values='销售量', columns='城市', aggfunc=sum, margins=True).reset_index().sort_values(by='All', ascending=True)
print(data4.to_string())
'''
城市    类别   上海    北京    广州    深圳  All
5      食品    NaN   10.0    NaN    NaN   10
2      电器    NaN    NaN    NaN   20.0   20
4      茶叶   80.0    NaN    NaN    NaN   80
3      皮具   50.0   60.0    NaN    NaN  110
0     化妆品   NaN    NaN   120.0   NaN  120
1    服装饰品  80.0   90.0    NaN  200.0  370
6      All   210.0  160.0  120.0  220.0  710
'''
# 5）计算排名
# 计算每个城市各种类商品总销量，并根据类别总销量排名
data5 = df.pivot_table(index=['城市', '类别'], values='销售量', aggfunc=sum).reset_index()
data5['rk'] = data5.groupby(by=['城市'])['销售量'].rank(method='first', ascending=False)
print(data5.to_string())
'''
   城市    类别  销售量   rk
0  上海  服装饰品  80   1.0
1  上海    皮具   50   3.0
2  上海    茶叶   80   2.0
3  北京  服装饰品  90   1.0
4  北京    皮具   60   2.0
5  北京    食品   10   3.0
6  广州   化妆品  120  1.0
7  深圳  服装饰品  200  1.0
8  深圳    电器   20   2.0
'''



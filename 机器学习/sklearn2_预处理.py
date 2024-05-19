"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2024/1/8
Time: 22:08
Description:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pumpkins = pd.read_csv('机器学习系列/US-pumpkins.csv')
# 查看数据
# print(pumpkins.head(100).to_string())
# 行列
# print(pumpkins.shape)
# 基本信息
# pumpkins.info()

# 通过观察，我们可以发现：
# 数据存在缺失值
# 数据共有1757行，代表1757个样本，按城市分组
# 数据的列是我们的特征变量

# 数据检查
# 缺失值检查
# 检查每列数据中的缺失值的数量
# print(pumpkins.isnull().sum())
# 其中City Name、Package、Variety、Date、Low Price、High Price、Origin、Repack列数据基本无缺失，其它列数据缺失较多

# 一致性检查
# 另外，我们发现南瓜的包装方式（称量单位）不统一
# print(pumpkins["Package"].is_unique)
# print(set(pumpkins["Package"].tolist()))

# 接下来，我们需要对数据做一些整理，这涉及到数据挖掘。我们需要从中提取一些有价值的数据，哪些数据有价值呢？

# 想一想，我们的目标是预测南瓜价格，那么价格是我们的因变量标签。而我们认为南瓜的价格可能与以下因素有关：
# 月份、销售日期、销售地区、南瓜种类、包装（称量单位）等

# 看一下我们的数据，这些特征有的已经存在，有的需要我们从已有的数据中提取
# 月份：可以从Date列中提取
# 销售日期：可以通过Date列转化得到（为方便绘图，需要统一转化为该年中的第几天）
# 销售地区：已存在的特征（City Name列）
# 南瓜种类：已存在的特征（Variety列，Sub Variety列缺失值较多不使用该列）
# 称量单位：已存在的特征（Package列）
# 南瓜价格：可以从已存在的特征（Low Price和High Price）计算平均值得到

# 数据整理

# 提取我们本次研究所需要的有价值的特征和标签
features = ['Date', 'City Name', 'Variety', 'Package', 'Low Price', 'High Price']
data = pumpkins[features]
# print(len(data))
# print(data.head().to_string())

# 过滤含有空值的数据
data.dropna(axis=0, how='any', inplace=True)
# print(len(data))

# 上面我们已经验证南瓜的包装方式（称量单位）不统一。经研究原始数据，我们发现其中包含“英寸”、“磅”和“蒲式耳”三种称重类型，单位的不同，南瓜似乎很难保持一致的重量，那么它的价格预测也毫无意义
# 因此，我们打算只筛选“蒲式耳”单位（只包含字符串“bushel”）的数据，其它的南瓜数据我们将过滤它们，虽然缺失了很多数据，但它们对分析无用
data = data[data['Package'].str.contains('bushel', case=True, regex=True)]
# print(len(data))
# print(data.head().to_string())

# 数据提取
# 提取月份
data['Month'] = data['Date'].apply(lambda dt: pd.to_datetime(dt).month)
# print(data.head().to_string())

# 销售日期转化为该年中的第几天
data['DayOfYear'] = data['Date'].apply(lambda dt: pd.to_datetime(dt).timetuple().tm_yday)
# print(data.head().to_string())

# 计算南瓜的平均价格作为标签
data['Price'] = (data['Low Price'] + data['High Price']) / 2
# print(data.head().to_string())

# 根据Package称量单位换算价格Price，转化为每bushel的价格（标准化）
# 1 1/9 bushel cartons => Price = Price/(1 + 1/9)
data.loc[data['Package'].str.contains('1 1/9'), 'Price'] = data['Price']/(1 + 1/9)
# 1/2 bushel cartons => Price = Price/(1/2)
data.loc[data['Package'].str.contains('1/2'), 'Price'] = data['Price']/(1/2)
# print(data.head(30).to_string())

# 根据The Spruce Eats的说法，bushel的重量取决于农产品的类型，因为它是体积测量。例如，1 bushel的西红柿应该重达56磅；叶子和绿色蔬菜以更轻的重量占用更多的空间，所以1 bushel的菠菜只有20磅
# 现在，我们可以根据测量值分析每单位的南瓜定价
# 你有没有注意到半bushel出售的南瓜非常昂贵？你能弄清楚这是为什么吗？因为小南瓜比大南瓜贵得多，可能是因为每bushel的小南瓜数量更多，因为一个大空心的南瓜占用了更多不必要的空间

# 将整理的数据放到新的DataFrame中
new_features = ['Month', 'DayOfYear', 'City Name', 'Variety', 'Package', 'Price']
new_pumpkins = data[new_features].reset_index(drop='index')
print(len(new_pumpkins))    # 共415条数据
print(new_pumpkins.head().to_string())

# 保存
# new_pumpkins.to_csv(r'C:\Users\cc\Desktop\new_pumpkins.csv', index=False, encoding='utf-8')

# 问题提出
# 在开始之前，我们先提出以下问题：
# 什么时候是购买南瓜的最佳时间？
# 如何预测一箱迷你南瓜的价格？
# 我们应该用1/2 bushel的篮子买它们，还是用10/9 bushel的篮子买？

# 相关性分析
# 月份与价格
# 绘制每个月南瓜的平均价格柱状图
# new_pumpkins.groupby(by='Month')['Price'].mean().plot(kind='bar')
# plt.ylabel('Price')
# plt.show()

# 从柱状图中可以直观的看出，南瓜的最高价格可能出现在9月或10月

# 城市与价格
# 绘制每个城市的南瓜平均价格柱状图
# new_pumpkins.groupby(by='City Name')['Price'].mean().plot(kind='bar')
# plt.xticks(rotation=30, ha='center')
# plt.ylabel('Price')
# plt.show()

# 可以看到，大部分城市南瓜的价格处于中等水平，少部分城市南瓜的价格相差较大

# 销售日期与价格
# 销售日期与价格散点图
# plt.scatter('DayOfYear', 'Price', data=new_pumpkins)
# plt.xlabel('DayOfYear')
# plt.ylabel('Price')
# plt.show()

# 这看起来似乎不同的价格群对应于不同的南瓜品种。为了证实这一点，我们使用不同的颜色标记不同的南瓜种类
# 销售日期与不同种类的南瓜价格散点图
ax = None
colors = ['red', 'blue', 'green', 'yellow']
for idx, var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety'] == var]
    ax = df.plot.scatter('DayOfYear', 'Price', ax=ax, c=colors[idx], label=var)

# plt.show()

# 根据上图，我们可以看到南瓜种类对南瓜价格的影响比销售日期更大。MINIATURE种类的南瓜价格整体较贵，而PIE TYPE种类的南瓜价格整体较便宜


# 在训练线性回归模型之前，确保数据是清理过的，这很重要！线性回归不能很好地处理缺失值，因此需要删除包含缺失值的样本
# 删除包含空值的行
new_pumpkins.dropna(inplace=True)
print(len(new_pumpkins))    # 415

# 分类特征变量的处理
# 我们希望能够使用相同的模型预测不同南瓜品种的价格。但是，南瓜类型列与其他特征略有不同，因为它的值是字符串而非数值
# 为了方便训练，我们首先需要将其转换为数值形式，对其进行编码：
# get_dummies()函数将用4个不同的列替换原有列，每个列对应一个品种。每列将包含相应种类的值，这意味着线性回归中将有四个系数，每个南瓜品种一个，负责该特定品种的起始价格
# X = pd.get_dummies(new_pumpkins['Variety'])
# print(X.info())
# 南瓜数据被分为4类，每类包含415个样本



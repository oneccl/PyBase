
# 标准化

from sklearn.preprocessing import StandardScaler

# # 初始化对特征和目标值的标准化器
# ss = StandardScaler()
# # 标准化处理
# ss.fit(X)
# # 转换数据为均值0、标准差1（标准化处理）
# # X为2D数组，形状(n_samples,n_features)，返回转换后形状相同的数组
# X_ss = ss.transform(X)
#
# # 简化写法
# # 标准化处理
# X_ss = ss.fit_transform(X)
#
# # 获取转换后数据的属性均值和标准差（方差）
# print(f'每列平均值：{ss.mean_}')
# print(f'每列标准差：{ss.var_}')
# # 也可以使用如下方式获取
# print(X_ss.mean())
# print(X_ss.std())


# 划分训练集和测试集

# from sklearn.model_selection import train_test_split
#
# # 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # 初始化对特征和目标值的标准化器
# ss = StandardScaler()
# # 分别对训练和测试数据的特征进行标准化处理
# X_train = ss.fit_transform(X_train)
# X_test = ss.fit_transform(X_test)


# 归一化

# from sklearn.preprocessing import MinMaxScaler
#
# # 初始化转换器（feature_range是归一化的范围，即最小值~最大值，默认0~1）
# transfer = MinMaxScaler(feature_range=(0, 1))
# # ⽣成min(X)和max(X)
# transfer.fit(X)
# # 最小-最大标准化
# # X为2D数组，形状(n_samples,n_features)，返回转换后形状相同的数组
# X_ms = transfer.transform(X)
#
# # 简化写法
# # 归一化处理
# X_ms = transfer.fit_transform(X)

# 划分训练集和测试集

# from sklearn.model_selection import train_test_split
#
# # 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # 初始化对特征和目标值的标准化器
# transfer = MinMaxScaler()
# # 分别对训练和测试数据的特征进行归一化处理
# X_train = transfer.fit_transform(X_train)
# X_test = transfer.fit_transform(X_test)


# sklearn.preprocessing中的标准化StandardScaler与scale的区别

# 标准化主要用于对样本数据在不同特征维度进行伸缩变换，目的是使得不同度量之间的特征具有可比性，同时不改变原始数据的分布
# 一些机器学习算法对输入数据的规模和量纲非常敏感，如果输入数据的特征之间存在数量级差异，可能会影响算法的准确性和性能
# 标准化处理的好处是我们在进行特征提取时，可以忽略不同特征之间由于噪声所导致的度量差异，而保留样本在各个维度上的信息分布，提高算法的准确性和性能，增加数据的可解释性

# 标准化的过程如下：
# - 计算数据列的算数平均值（mean）
# - 计算数据列的标准差/方差（std）
# - 对每个数据列分别进行转化：(X-mean)/std

# sklearn.preprocessing提供了两种直接对给定数据进行标准化的方式：scale()函数和StandardScaler类
# 它们之间有什么区别呢？

import numpy as np
from sklearn.preprocessing import scale, StandardScaler

# # A、scale(X, axis)函数：axis：用来计算均值和标准差的轴，默认0，对每个特征进行标准化（列），1为对每个样本进行标准化（行）
# # 样本数据
# X = np.array([[1, -1, 2], [2, 1, 0]])
# # 直接标准化处理
# X_scaled = scale(X)
# print(X_scaled)
# '''
# [[-1. -1.  1.]
#  [ 1.  1. -1.]]
# '''
# # 处理后数据的均值和方差
# print(X_scaled.mean(axis=0))    # [0. 0. 0.]
# print(X_scaled.std(axis=0))     # [1. 1. 1.]
#
# # B、StandardScaler类
# ss = StandardScaler()
# # 标准化处理，如果在训练集上进行标准化，同时可以使用保存在训练集中的参数（均值、方差）对测试集数据进行转化
# X_scaled = ss.fit_transform(X)
# print(X_scaled)
# '''
# [[-1. -1.  1.]
#  [ 1.  1. -1.]]
# '''
# # 处理后数据的均值和方差
# print(X_scaled.mean())    # 0.0
# print(X_scaled.std())     # 1.0
# # 使用训练集标准化后的均值和方差对测试集数据进行转换
# print(ss.transform([[-1, 2, 0]]))
# '''
# [[-5.  2. -1.]]
# '''
#
# # 区别
# # - scale()函数：不能将原数据集（训练集）的均值和方差应用到新的数据集（测试集），如果使用全部样本，标准化计算的结果是训练集和测试集共同的期望和方差
# # - StandardScaler类：可以将原数据集（训练集）的均值和方差应用到新的数据集（测试集），即假设训练集的期望和测试集的期望是一样的，测试集的标准化是用的训练集的期望和方差
#
# # 在机器学习中，我们通常是从整体中以抽样的方式抽出训练集，这意味着我们默认这部分训练集可以代替整体，也就是训练集的期望就是整体的期望，测试集标准化时，它的期望采用的正是训练集的期望，所以StandardScaler()才是我们经常用的方式



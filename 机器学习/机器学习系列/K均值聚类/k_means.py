

# Scikit-Learn K均值聚类

# K-均值（K-Means）是一种聚类算法，属于无监督学习。K-Means在机器学习知识结构中的位置如下：

# 聚类（Clustering）是指将一个数据对象集合划分成簇（子集），使得簇内对象彼此相似，簇间对象不相似。通俗来说，就是将数据划分到不同组中
# 根据算法原理，常用的聚类算法可分为：基于划分的聚类算法K-Means、基于层次的聚类算法HC、基于密度的聚类算法。本文主要介绍K-Means聚类
# K-Means算法起源于1967年，由James MacQueen和J.B.Hartigan提出。K-Means中的K指的是类的数量，Means指均值
# K-Means算法的基本原理是：根据样本特征的相似度或距离远近，将样本（N个点）划分成若干个类（K个集群），使得每个点都属于离其最近的中心点（均值）对应的类（集群）
# 其中，相似度通常使用欧几里得距离来度量，用于计算数据点与质心之间的距离（使用平方）：
# $$d(X_i,C_j)=||X_i-C_j||^2$$
# 其中，$X_i$是数据点，$C_j$是质心
# K-Means假设一个样本属于一个类，K-Means的类别是样本的中心（均值）；K-Means的损失函数是样本与其所属类的中心之间的距离平方和：
# $$J=\sum_{j=1}^{k}\sum_{i=1}^{N_j}||X_i-C_j||^2$$
# 其中，$N_j$表示第$j$个簇中的样本数量

# K-Means算法的本质是物以类聚，其主要执行步骤如下：
# - 初始化聚类中心（随机选取K个样本作为初始的聚类中心）
# - 给聚类中心分配样本（计算各样本与各聚类中心的距离，把每个样本分配给距离它最近的聚类中心）
# - 移动聚类中心（新的聚类中心移动到这个聚类所有样本的均值处）
# - 停止移动（重复第二、第三步，直到聚类中心不再移动为止）
# K-Means算法采用的是迭代的方法，得到的是局部最优解
# 那么，如何确定K值呢？K-Means通常根据损失函数和轮廓系数确定K值，$J$越小，轮廓系数越大，聚类效果越好

# Scikit-Learn K均值聚类API
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://scikit-learn.org.cn/view/383.html
# API参数及说明如下：
# |`n_clusters`|要形成的簇数或要生成的质心数，默认为8
# |`init`|初始化方法，默认为`k-means++`，表示选择初始的聚类中心之间的距离要尽可能远；其他还有`random`：从初始质心的数据中随机选择观测值
# |`n_init`|使用不同质心运行K均值算法的次数种子，默认为`auto`
# |`max_iter`|k均值算法的最大迭代次数，默认300
# |`tol`| 收敛阈值，默认为`1e-4`
# |`random_state`|确定质心初始化的随机数生成，默认为None
# |`copy_x`| 是否复制原始数据，默认为True
# |`algorithm`|K-Means要使用的算法，默认为`auto`，其他参数有`full`、`elkan`
# 常用属性及说明如下：
# |`cluster_centers_`|簇中心坐标
# |`labels_`|每个点的分类标签
# |`inertia_`|样本到其最近的聚类中心的距离平方和
# |`n_iter_`|迭代次数
# 常用方法及说明如下：
# |`fit(X,y)`| 训练K-均值聚类
# |`fit_transform(X)`|训练K-均值聚类并将X变换为簇距离空间
# |`predict(X)`|预测X中每个样本所属的最接近的聚类
# |`transform(X)`|将X转换为簇距离空间

import pandas as pd
import numpy as np

# 下面使用样本数据进行演示
# 创建样本数据
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# # 生成了包含5个类别的1000条样本数据
# X, y = make_blobs(n_samples=1000, centers=5, random_state=1)
# plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=15)
#
# plt.show()

# 这里我们指定聚类的数目为5，但是实际情况中我们是不知道聚类的数目是多少的，这需要多次尝试

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score   # 轮廓系数法SC

# # 划分训练集（80%）和测试集（20%）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # K-Means聚类器
# kmeans = KMeans(n_clusters=5, n_init="auto")
#
# # 训练模型
# kmeans.fit(X_train, y_train)
#
# # # 5组数据的中心点
# # print(kmeans.cluster_centers_)
# # # 每个数据点所属分组（从0~5）
# # print(kmeans.labels_)
#
# # 模型评估
# # 损失函数J（拐点法/手肘法）
# print(kmeans.inertia_)   # 1525.6665836724683
# # 轮廓系数法SC
# print(silhouette_score(X_test, y_test))   # 0.5635285557582774
#
# # 绘制质心
# from pylab import mpl
#
# # 使用全局字体
# plt.rcParams["font.family"] = "SimHei"
# # 设置正常显示符号
# mpl.rcParams["axes.unicode_minus"] = False
#
# # 创建一个画板和两个子图（1x2）
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# markers = ["x", "o", "^", "s", "*"]
# centers = kmeans.cluster_centers_
#
# axes[0].scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, s=15)
# axes[0].set_title("训练集的质心位置")
#
# axes[1].scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test, s=15)
# axes[1].set_title("测试集的质心位置")
#
# for idx, c in enumerate(centers):
#     axes[0].plot(c[0], c[1], markers[idx], markersize=10)
#     axes[1].plot(c[0], c[1], markers[idx], markersize=10)
#
# plt.show()


# # K-Means最佳K
# from sklearn.datasets import make_blobs
#
# # 生成了包含5个类别的1000条样本数据
# X, y = make_blobs(n_samples=1000, centers=5, random_state=1)
#
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score   # 轮廓系数法SC
#
# # 最佳K
# # 手肘法（误差平方和）
# distortions = []
# for k in range(2, 10):
#     # K-Means聚类器
#     kmeans = KMeans(n_clusters=k, n_init="auto")
#     # 训练模型
#     kmeans.fit(X, y)
#     # SSE
#     distortions.append(kmeans.inertia_)
#
# # 绘制手肘图
# plt.plot(range(2, 10), distortions, marker='x')
# plt.xlabel('K')
# plt.ylabel('Distortion')
# plt.title('Distortion-K')
# plt.show()
#
# # 轮廓系数法（轮廓系数）
# # 好的聚类，内密外疏，同一个聚类内部的样本要足够密集，不同聚类之间样本要足够疏远
# # 轮廓系数是聚类的重要评估指标。轮廓系数综合考虑了内聚度和分离度两种因素
# # 轮廓系数使用每个样本的平均集群内距离（a）和平均最近集群距离（b）进行计算，具体计算规则为：针对样本空间中的一个特定样本，计算它与所在聚类其它样本的平均距离（a）和该样本与距离最近的另一个聚类中所有样本的平均距离（b），该样本的轮廓系数为
# # $$S(i)=\frac{b(i)-a(i)}{max(a(i),b(i))}$$
# # 将整个样本空间中所有样本的轮廓系数取算数平均值，作为聚类的性能指标SC
# # 轮廓系数的区间为：`[-1,1]`。-1代表分类效果差，1代表分类效果好，0代表聚类重叠，负值则表示样本分配给了错误的聚类
# # Scikit-Learn提供了轮廓系数API：
# # sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
# # API中文文档参考：https://scikit-learn.org.cn/view/539.html
#
# scores = []
# for k in range(2, 10):
#     # K-Means聚类器
#     kmeans = KMeans(n_clusters=k, n_init="auto")
#     # 训练模型并预测
#     labels = kmeans.fit_predict(X, y)
#     # SC
#     scores.append(silhouette_score(X, labels))
#
# # 轮廓系数法
# plt.plot(range(2, 10), scores, marker='x')
# plt.xlabel('K')
# plt.ylabel('Score')
# plt.title('Score-K')
# plt.show()


# 优缺点
# 优点：
# - 超参数少，只需要一个K值，易于理解和使用，预测效果较好，特别是当簇近似高斯分布时
# - 可解释性强，收敛速度快，可用于大型数据集
# 缺点：
# - K的选取困难，对初始聚类中心敏感，不同的K得到的结果不同
# - 对于非凸的数据集较难收敛，聚类效果不佳
# - 对于噪音和异常点敏感，离群点或噪声数据会对均值产生较大影响，导致中心偏移

# 聚类与分类的区别
# - 聚类属于无监督学习，分类属于监督学习
# - 聚类问题的类别是未知的（目标不明确），分类问题的类别是已知的（目标明确）
# - 聚类的效果很难评估，分类的效果容易评估



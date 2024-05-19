
# 常用机器学习算法与实践

"""
一、监督学习算法：回归、决策树、随机森林等
在监督学习中，我们有一个标签或结果，模型根据提供的输入-输出对进行训练，以便在给定新输入时能做出准确的预测
"""
# 1、线性回归
'''
线性回归是一种预测模型，用于找出输入变量和输出变量之间的线性关系。如果有多个输入变量，称为多元线性回归
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 2、决策树
'''
决策树是一种决策支持工具，它使用树状模型的决策方式，同时也是一种常见的机器学习方法
'''
from sklearn.tree import DecisionTreeClassifier

# 创建模型
model = DecisionTreeClassifier()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 3、随机森林
'''
随机森林是一种集成学习模型，它会构建多个决策树，并取它们的平均或多数投票结果
'''
from sklearn.ensemble import RandomForestClassifier

# 创建模型
model = RandomForestClassifier()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

"""
二、无监督学习算法：聚类、降维等
在无监督学习中，我们没有标签或结果，模型需要从输入数据中发现结构或关系
"""
# 4、K-means聚类
'''
K-means 是一种简单而流行的聚类算法。给定一个聚类数量 K，它会尝试将数据分割成K个不相交
的子集，以使得同一子集中的数据点彼此之间尽可能相似，而不同子集的数据点尽可能不同
'''
from sklearn.cluster import KMeans

# 创建模型
kmeans = KMeans(n_clusters=3)

# 训练模型
kmeans.fit(X)

# 预测
y_pred = kmeans.predict(X)

# 5、主成分分析（PCA）
'''
PCA 是一种常见的降维方法。它通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量
'''
from sklearn.decomposition import PCA

# 创建模型
pca = PCA(n_components=2)

# 训练模型
X_pca = pca.fit_transform(X)

"""
三、强化学习算法简介
强化学习是机器学习的一个子领域，其中一个智能体学习如何在环境中采取行动，以最大化某种累积奖励。Q-learning 是一种简单而有效的强化学习方法
"""

# 机器学习算法的选择与应用
'''
选择哪种机器学习算法通常取决于你的数据和问题。一般来说，如果你有标签数据，并且是预测问题，你可以选择监督学习
如果你只有输入数据没有标签，你可以选择无监督学习。如果你的问题是关于时序决策的，那么可能需要强化学习
此外，需要考虑的其他因素包括数据的大小、维度、是否线性可分、是否有缺失值等等
在实践中，常常需要尝试多种方法，通过交叉验证来评估模型的性能，然后选择最好的一种
'''
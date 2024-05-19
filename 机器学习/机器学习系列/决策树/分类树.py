
# https://www.showmeai.tech/
# https://www.showmeai.tech/article-detail/190

# Scikit-Learn决策树
# 决策树分类

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def decision_boundary_fill(model, axis):
    # np.meshgrid()：定义X/Y坐标轴上的起始点和结束点以及点的密度，返回这些网格点的X和Y坐标矩阵
    X0, X1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    # ravel()：将高维数组降为一维数组
    # np.c_[]：将两个数组以列的形式拼接起来形成矩阵，这里将上面每个网格点的X和Y坐标组合
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]
    # 通过训练好的逻辑回归模型，预测平面上这些网格点的分类
    y_grid_pred = model.predict(X_grid_matrix)
    y_pred_matrix = y_grid_pred.reshape(X0.shape)
    # plt.contourf(X,Y,Z)：绘制等高线，cmap用于设置填充轮廓，默认为viridis，还可设置为热力图色彩plt.cm.hot或自定义；alpha用于设置填充透明度
    # ListedColormap()：自定义填充色彩列表
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(X0, X1, y_pred_matrix, alpha=0.5, cmap=custom_cmap)


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
#
# # 生成两个交错的半圆形状数据集
# X, y = datasets.make_moons(noise=0.25, random_state=666)
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()
#
# from sklearn.tree import DecisionTreeClassifier      # 决策树分类器
#
# # 使用CART分类树的默认参数
# # dt_clf = DecisionTreeClassifier()
# dt_clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=4)
# # 训练拟合
# dt_clf.fit(X, y)
# # 绘制决策边界
# decision_boundary_fill(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()

# 葡萄酒数据集
# 葡萄酒（Wine）数据集是来自加州大学欧文分校（UCI）的公开数据集，这些数据是对意大利同一地区种植的葡萄酒进行化学分析的结果。数据集共178个样本，包括三个不同品种，每个品种的葡萄酒中含有13种成分（特征）、一个类别标签，分别使是0/1/2来代表葡萄酒的三个分类

from sklearn.datasets import load_wine

wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['class'] = wine.target
# print(data.head().to_string())
# | 属性/标签 | 说明  |
# |--|--|
# | `alcohol`  |  酒精含量（百分比）  |
# | `malic_acid`  |  苹果酸含量（克/升）  |
# | `ash`  |  灰分含量（克/升）  |
# | `alcalinity_of_ash` |  灰分碱度（mEq/L）  |
# | `magnesium` | 镁含量（毫克/升）  |
# | `total_phenols` | 总酚含量（毫克/升）  |
# | `flavanoids` | 类黄酮含量（毫克/升）  |
# | `nonflavanoid_phenols` | 非黄酮酚含量（毫克/升） |
# | `proanthocyanins` |  原花青素含量（毫克/升）  |
# | `color_intensity` |  颜色强度（单位absorbance） |
# | `hue`  | 色调（在1至10之间的一个数字） |
# |`od280/od315_of_diluted_wines` |  稀释葡萄酒样品的光密度比值，用于测量葡萄酒中各种化合物的浓度  |
# | `proline` |  脯氨酸含量（毫克/升） |
# | `class` |  分类标签（class_0(59)、class_1(71)、class_2(48)） |

# # 数据集大小
# print(wine.data.shape)      # (178, 13)
# # 标签名称
# print(wine.target_names)    # ['class_0' 'class_1' 'class_2']
# # 分类标签
# print(data.groupby('class')['class'].count())
# '''
# class
# 0    59
# 1    71
# 2    48
# '''
# print('=======')
# # 缺失值：无缺失值
# print(data.isnull().sum())

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# https://scikit-learn.org.cn/view/784.html

from sklearn.model_selection import train_test_split     # 数据集划分
from sklearn.tree import DecisionTreeClassifier          # 决策树模型
from sklearn import tree

# 特征
X = wine.data
# 标签
y = wine.target

# 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # 决策树分类器（CART）
# # clf = DecisionTreeClassifier(criterion='gini', random_state=0)
# clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, min_samples_split=5, random_state=0)
# # 训练模型
# clf.fit(X_train, y_train)
# # 在测试集上预测
# y_pred = clf.predict(X_test)
# print(y_test)  # [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0]
# print(y_pred)  # [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 0 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0]
#
# # 模型评估
# # 平均准确度
# print(clf.score(X_test, y_test))    # 0.9722222222222222
# # max_depth=3, min_samples_leaf=5, min_samples_split=5：0.8611111111111112
#
# # 模型的参数与特征重要性
# # 决策树深度
# print(clf.get_depth())      # 4
# # 决策树叶子数
# print(clf.get_n_leaves())   # 6
# # 特征的重要程度
# importance = pd.DataFrame()
# importance['features'] = wine.feature_names
# importance['importance'] = clf.feature_importances_
# importance.sort_values('importance', ascending=False, inplace=True, ignore_index=True)
# print(importance.to_string())
# # 绘制决策树（二叉树）
# # plot_tree(decision_tree,filled,rounded)
# # - class_names：是否显示分类，布尔类型默认使用0、1、...；也可使用数组自定义
# # - decision_tree：决策树模型
# # - filled：是否填充颜色
# # - rounded：是否使用圆角框
# tree.plot_tree(clf, class_names=True, filled=True, rounded=True)
# plt.show()


# https://blog.csdn.net/weixin_41885239/article/details/121870123
# https://blog.csdn.net/weixin_60476982/article/details/136627215
# 参数调优与选择
# 网格搜索是一种穷举搜索的调参手段：遍历所有的候选参数，循环建立模型并对模型的准确性进行评估，选取表现最好的参数作为最终结果
# 如果要同时调节多个模型参数，例如，模型有两个参数，第一个参数有3种取值可能，第二个参数有4种取值可能，则所有的可能性列举出来可以表示成3*4的网格，遍历的过程像是在网格（Grid）中搜索（Search），因此该方法被称为网格搜索
# 如果设置了交叉验证参数cv，例如cv=5，则每种参数组合的模型都会执行5遍（5折），该参数组合的模型评分取值为这5遍的平均评分

from sklearn.model_selection import GridSearchCV

# 模型候选参数
params = {
    'criterion': ['gini', 'entropy'],    # 最优划分策略
    'max_depth': range(3, 10),           # 限制决策树的最大深度
    'min_samples_leaf': range(2, 10),    # 将样本数量小于min_samples_leaf的叶子节点剪掉
    "min_samples_split": range(2, 15)    # 将中间节点样本数量小于min_samples_split的剪掉
}
# 网格搜索：搜索所有组合，评估每种组合
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    param_grid=params,
    scoring='accuracy',
    cv=5
)
# 拟合所有组合
grid.fit(X_train, y_train)

# 获取最佳参数
# 最优评分
print(grid.best_score_)      # 0.9362068965517241
# 最佳参数
print(grid.best_params_)     # {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 2}
# 获取最佳评分时的模型
print(grid.best_estimator_)  # DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0)

from sklearn.metrics import accuracy_score

# 基于最佳参数建模
best_clf = grid.best_estimator_
# 训练模型
best_clf.fit(X_train, y_train)
# 在测试集上预测
y_pred = best_clf.predict(X_test)
# 准确度评分
print(accuracy_score(y_test, y_pred))  # 0.9722222222222222

# 绘制决策树
tree.plot_tree(best_clf, class_names=True, filled=True, rounded=True)
plt.show()

# 上面交叉验证默认使用准确度作为评价标准，如果想以ROC曲线的AUC值作为评分标准，可以设置`scoring`参数为`roc_auc`



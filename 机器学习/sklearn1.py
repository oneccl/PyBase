
# Scikit-Learn库
# 中文：https://scikit-learn.org.cn/
# 英文：https://scikit-learn.org/stable/index.html

# Scikit-learn是一个Python机器学习库，广泛地应用于统计分析和机器学习建模等数据科学领域
# sklearn全称scikit-learn，建立在numpy、scipy、matplotlib等数据科学包的基础之上，涵盖了机器学习中的样例数据、数据预处理、模型验证、特征选择、分类、回归、聚类、降维等几乎所有环节
# 特点：
"""
1）数据建模：Scikit-learn可以实线各种监督和非监督学习方模型
2）功能多样：Scikit-learn可以对数据进行预处理、特征工程、数据集切分、模型评估等
3）数据丰富：Scikit-learn内置丰富的数据集
"""

# Scikit-Learn库的数据集
# 在机器学习中，经常需要使用各种各样的数据集，Scikit-Learn库提供了一些常用的数据集，用于快速搭建机器学习任务、对比模型性能
"""
数据集	          调用方法  	             描述
鸢尾花数据集	      Load_iris()	         特征为连续数值变量，标签为0/1/2的三分类任务，且各类样本数量均衡，均为50个，用于多分类任务的数据集
波士顿房价数据集	  Load_boston()	         连续特征拟合房价，用于回归任务的经典数据集（自1.2版本已移除）
糖尿病数据集	      Load_diabetes()	     用于回归任务的经典数据集
毛写数字数据集	  Load_digits()	         小型手写数字数据集，包含0-9共10种标签，各类样本均衡，特征是离散数，用于多分类任务的数据集
乳腺癌数据集	      Load_breast_cancer()	 特征为连续数值变量，标签为0或1的二分类任务，用于二分类任务的数据集
体能训练数据集	  Load_linnerud()	     经典的用于多变量回归任务的数据集
红酒数据集         Load_wine()            各类样本数量轻微不均衡，用于连续特征的3分类任务
"""

# Scikit-Learn库的功能
# 1）分类：
# 分类是指识别给定对象的所属类别，属于检测学习的范畴，最常见的应用场景包括垃圾邮件检测和图像识别等
# 目前Scikit-learn已经实现的算法包括：支持向量机（SVM），最近邻、逻辑回归、随机森林、决策树以及多层感知器（MLP）神经网络等
# 2）回归：
# 回归是指预测与给定对象相关联的连续值属性，最常用的应用场景包括预测药物反应和预测股票价格等
# 目前Scikit-Learn已经实现的算法包括：支持向量回归（SVR），脊回归，Lasso回归，弹性网络（ElasticNet），最小角回归（LARS），贝叶斯回归，以及各种不同的鲁棒回归算法等
# 3）聚类：
# 聚类是指自动识别具有相似属性的给定对象，并将其分组为集合，数据无监督学习范畴，最常见的应用场景包括顾客细分和实验结果分组等
# 目前Scikit-Learn已经实现的算法包括：K-均值聚类、谱聚类、均值偏移、分层聚类、DBSCAN聚类等
# 4）数据降维：
# 数据降维是指使用主成分分析（PCA），非矩阵分解（NMF）或特征选择等降维技术来减少要考虑的随机变量的个数，其主要应用场景包括可视化处理和效率提升
# 目前Scikit-Learn已经实现的算法包括：K-均值、特征选择、非负矩阵分解等
# 5）模型选择：
# 数据降维是指使用主成分分析（PCA），非矩阵分解（NMF）或特征选择等降维技术来减少要考虑的随机变量的个数，其主要应用场景包括可视化处理和效率提升
# 目前Scikit-Learn实现的模块包括：格点搜索、交叉验证和各种针对预测误差评估的度量函数
# 6）数据预处理：
# 数据预处理是指数据的特征提取和归一化，是机器学习过程中的第一个也是最重要的一个环节
# 归一化是指将输入数据转化为具有零均值（中心化）和单位权方差的新变量，但因为大多数时候都做不到精确等于零，因此会设置一个可接受的范围，一般都要求落在0-1之间
# 特征提取是指将文本或图像数据转换为可用于机器学习的数据变量

# 安装：
# pip install scikit-learn

import numpy as np
import pandas as pd
import sklearn
# 导入数据集
from sklearn import datasets

# 分类数据：iris数据
iris = datasets.load_iris()
# print(type(iris))          # <class 'sklearn.utils._bunch.Bunch'>

# print(iris)                # 详细信息
# 数据集的属性
# print(iris.data)              # 数据集样本数据
# print(iris.target)            # 数据集标签数据
# print(iris.target_names)      # 数据集标签
# print(iris.feature_names)     # 数据集特征
# print(iris.filename)          # 数据集文件路径

# 其他
# print(iris.keys())                          # 查看键（属性）
# print(iris.data.shape, iris.target.shape)   # 查看数据形状
# print(iris.DESCR)                           # 数据集描述信息

# 生成DataFrame
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df_iris.to_string())

# 添加因变量
df_iris['target'] = iris.target
# print(df_iris.to_string())

# 生成数据
# 方式1：
# data_X = iris.data        # 导入样本数据
# data_y = iris.target      # 导入标签数据
# 方式2：直接返回
data_X, data_y = datasets.load_iris(return_X_y=True)

# 模型选择
# 训练集和测试集切分
from sklearn.model_selection import train_test_split

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    data_X,
    data_y,
    test_size=0.2,      # 切分比例
    random_state=111
)
# print(len(X_train))       # 150*(1-0.2)=120

# 数据预处理
# 数据标准化和归一化
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化：适用于可能存在极大或极小异常值，可能会因单个异常点而将其他数值变换的过于集中；处理后，均值为0方差为1
ss = StandardScaler()
X_scaled = ss.fit_transform(X_train)

# 归一化：适用于数据有明显的上下限，不会存在严重的异常值，如考试分数；处理后，在0-1之间
mm = MinMaxScaler()
X_scaled = mm.fit_transform(X_train)

# 类型编码
from sklearn.preprocessing import LabelEncoder

# 对数字编码
le = LabelEncoder()
data = [1, 2, 2, 6]
le.fit(data)
# print(le.classes_)      # [1 2 6]

# 每个元素出现的顺序
# print(le.transform(data))      # [0 1 1 2]

# 根据顺序获取对应元素（类似反编码）
# print(le.inverse_transform([1, 0, 2, 1]))      # [2 1 6 2]

# 对字符串编码
data = ['shenzhen', 'shenzhen', 'shanghai', 'beijing']
le.fit(data)
# print(le.classes_)      # ['beijing' 'shanghai' 'shenzhen']

# print(le.transform(data))      # [2 2 1 0]

# print(le.inverse_transform([0, 2, 1, 0]))     # ['beijing' 'shenzhen' 'shanghai' 'beijing']

# 分析案例

# 近邻模型：K近邻分类器（KNN）
# KNN：通过计算待分类的数据点，与已有数据集中所有数据点的距离，取距离最小的前K个点，根据少数服从多数的原则，将这个数据点划分为出现次数最多的那个类别
from sklearn.neighbors import KNeighborsClassifier         # K近邻分类器模型（KNN）
from sklearn.datasets import load_iris                     # 数据集
from sklearn.model_selection import train_test_split       # 切分数据
from sklearn.model_selection import GridSearchCV           # 模型选择：网格搜索
from sklearn.pipeline import Pipeline                      # 流水线管道操作
from sklearn.metrics import accuracy_score                 # 得分验证

# 模型实例化：创建一个K近邻分类器
'''主要参数
- n_neighbors：用于指定分类器中K的大小（默认为5）
- Weights：设置选中的K个点对分类结果影响的权重（默认为平均权重uniform），可以选择distance：表示越近的点权重越高，还可以传入自定义权重计算函数
- algorithm：设置用于计算临近点的方法，选项中有ball_tree、kd_tree和brute，代表不同寻找邻居的优化算法（默认为auto，根据训练数据自动选择）
'''
knn = KNeighborsClassifier()
# 训练模型
knn.fit(X_train, y_train)
# 测试集预测
y_pred = knn.predict(X_test)              # 基于模型的预测值
# print(y_pred)      # [0 0 2 2 1 0 0 2 2 1 2 0 1 2 2 0 2 1 0 2 1 2 1 1 2 0 0 2 0 2]
# 模型得分验证
# 方式1
# print(knn.score(X_test, y_test))          # 0.9333333333333333
# 方式2
# print(accuracy_score(y_pred, y_test))     # 0.9333333333333333


# 模型选择：网格搜索

# 搜索的参数
knn_paras = {'n_neighbors': [1, 3, 5, 7]}
# 默认的模型
knn_grid = KNeighborsClassifier()

# 网格搜索对象实例化
# 模型选择：K折交叉验证：10折交叉验证
# 默认K=5折，相当于把数据集平均切分为5份，并逐一选择其中一份作为测试集，其余作为训练集进行训练及评分，最后返回K个评分
grid_search = GridSearchCV(knn_grid, knn_paras, cv=10)
grid_search.fit(X_train, y_train)

# 通过搜索找到最好的参数值
print(grid_search.best_estimator_)     # KNeighborsClassifier(n_neighbors=7)
print(grid_search.best_params_)        # {'n_neighbors': 7}
print(grid_search.best_score_)         # 0.975

# 基于搜索结果建模
knn1 = KNeighborsClassifier(n_neighbors=7)
knn1.fit(X_train, y_train)

y_pred1 = knn1.predict(X_test)
print(knn1.score(X_test, y_test))         # 1.0
print(accuracy_score(y_pred1, y_test))    # 1.0

# 结论：网格搜索后的建模效果优于未使用网格搜索的模型

# 度量指标
# 分类任务：准确率accuracy、精准率precision/召回率recall、调和平均数F1
# 回归任务：均方差MSE、平均绝对差MAE、R2_score
# 聚类任务：轮廓系数silhouette_score、调整兰德指数adjusted_rand_score



# scikit-learn库简介与使用

"""
1、scikit-learn库的安装和配置
"""
'''
Scikit-learn是一个在Python编程语言中实现的免费软件机器学习库。它具有各种分类、回归和聚类算法
包括支持向量机、随机森林、梯度提升、k-means和DBSCAN，并设计为与Python数值和科学库NumPy和SciPy一起使用
'''
'''
要在Python环境中安装Scikit-learn，你可以使用pip（Python的包管理器）或者conda（Anaconda发行版的包和环境管理器）进行安装
1）使用pip安装：
pip install -U scikit-learn
2）使用conda安装：
conda install scikit-learn
'''

# 2、scikit-learn库的数据预处理方法
'''
Scikit-learn提供了许多数据预处理方法，包括：数据缩放、类别编码、缺失值处理
'''
# 1）数据缩放：例如StandardScaler进行z-score标准化，MinMaxScaler进行最小-最大标准化等
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2）类别编码：例如OneHotEncoder进行一位有效编码，LabelEncoder进行标签编码等
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# 3）缺失值处理：例如SimpleImputer可以用各种方法（例如平均值、中位数、最常见的值等）来填充缺失值
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 3、scikit-learn库的模型构建与训练
'''
在Scikit-learn中，每个模型都是一个Python类，可以用一组参数进行初始化。模型的训练通常由fit方法完成，预测通常由predict方法完成
'''
# Scikit-learn模型构建和训练的基本流程：

from sklearn.ensemble import RandomForestClassifier

# 创建模型：创建一个随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型：用训练数据进行训练
clf.fit(x_train, y_train)

# 预测：用测试数据进行预测
y_pred = clf.predict(X_test)


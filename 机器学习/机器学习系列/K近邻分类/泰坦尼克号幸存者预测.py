
import numpy as np
import pandas as pd

# A、泰坦尼克号幸存者数据分析

# 泰坦尼克号的沉没是世界上最严重的海难事故之一，造成了大量的人员伤亡。这是一艘号称当时世界上最大的邮轮，船上的人年龄各异，背景不同，有贵族豪门，也有平民旅人，邮轮撞击冰山后，船上的人马上采取措施安排救生艇转移人员，从本次海难中存活下来的，也就是幸存者

# 泰坦尼克号数据集为1912年泰坦尼克号沉船事件中相关人员的个人信息以及存活状况。包含了2224名乘客和船员的姓名、性别、年龄、船票等级、船票价格、船舱号、登船港口、生存情况等信息。这些历史数据已经被分为训练集和测试集，我们可以根据训练集训练出合适的模型并预测测试集中的存活状况
# 数据来源：https://www.kaggle.com/c/titanic
# - `gender_submission.csv`：乘客编号与是否幸存记录
# - `train.csv`：训练集
# - `test.csv`：测试集

# 数据集的属性信息（11特征+1标签）如下：
# | PassengerId | 乘客编号
# | Survived | 是否幸存，1是0否
# | Pclass | 船舱等级，1（一等）、2（二等）、3（三等）
# | Name | 乘客姓名
# | Sex | 乘客性别
# | Age | 乘客年龄
# | SibSp | 与乘客同行的兄弟姐妹及配偶人数
# | Parch | 与乘客同行的父母及子女人数
# | Ticket | 船票编号
# | Fare | 船票价格
# | Cabin | 乘客座位号
# | Embarked | 乘客登船码头，C（Cherbourg）、Q（Queenstown）、S（Southampton）

# 问题提出：哪些人可能成为幸存者？

# 1）数据集加载与概览
path = r"C:\Users\cc\Desktop\titanic_dataset\titanic_dataset\train.csv"
# 加载数据集
data = pd.read_csv(path, encoding='utf-8')
# # 数据集前5行
# print(data.head().to_string())
# '''
#    PassengerId  Survived  Pclass                                                 Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
# 0            1         0       3                              Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Thayer)  female  38.0      1      0          PC 17599  71.2833   C85        C
# 2            3         1       3                               Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
# 3            4         1       1         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
# 4            5         0       3                             Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
# '''
# # 数据集的大小
# print(data.shape)    # (891, 12)
#
# # 数据集的缺失情况
# print(data.isnull().sum())   # 乘客年龄和乘客座位号有大量缺失
#
# # 数据集的特征和标签
# data.info()
#
# # 分析
# # 获救人数占比
# survive_ratio = data['Survived'].value_counts(normalize=True).reset_index()
# print(survive_ratio)
# # 幸存者占比：38.4%；遇难者占比：61.6%
#
# # 性别特征对获救率的影响
# # 男性乘客与女性乘客占比
# mf_count = data['Sex'].value_counts().reset_index()
# print(mf_count)
# # 男性乘客：577人，女性乘客：314人
#
# # 男性乘客与女性乘客的获救率
# mf_ratio = data['Survived'].groupby(data['Sex']).value_counts().reset_index()
# # 合并
# mf_data = pd.merge(mf_ratio, mf_count, how='left', on='Sex')
# mf_data['rescue_ratio'] = mf_data['count_x'] / mf_data['count_y']
# print(mf_data)
# # 男性乘客获救比例：18.9%，女性乘客获救比例：74.2%
# # 女性乘客总人数比男性少，但是获救人数却比男性乘客要多。性别特征对获救概率影响较大
#
# # 船舱等级特征对获救率的影响
# # 各船舱等级乘客占比
# pc_count = data['Pclass'].value_counts().reset_index()
# print(pc_count)
# # 一等：216人，二等：184人，三等：491人
#
# # 各船舱等级乘客的获救率
# pc_ratio = data['Survived'].groupby(data['Pclass']).value_counts().reset_index()
# # 合并
# pc_data = pd.merge(pc_ratio, pc_count, how='left', on='Pclass')
# pc_data['rescue_ratio'] = pc_data['count_x'] / pc_data['count_y']
# print(pc_data)
# # 一等获救比例：62.9%，二等获救比例：47.3%，三等获救比例：24.2%
# # 一等船舱获救比例最高，三等船舱获救比例最低。船舱等级对于乘客的获救率存在较大的影响
#
# # 各船舱等级中的性别特征对获救率的影响
# # 不同船舱等级的男女乘客人数
# ps_count = data['Sex'].groupby(data['Pclass']).value_counts().reset_index()
# print(ps_count)
# # 一等：男122人，女94人，二等：男108人，女76人，三等：男347人，女144人
#
# # 不同等级船舱的男性乘客与女性乘客的获救率
# ps_ratio = data['Survived'].groupby([data['Pclass'], data['Sex']]).value_counts().reset_index()
# # 合并
# ps_data = pd.merge(ps_ratio, ps_count, how='left', on=['Pclass', 'Sex'])
# ps_data['rescue_ratio'] = ps_data['count_x'] / ps_data['count_y']
# print(ps_data)
# # 一等获救比例：男39.9%，女96.8%，二等获救比例：男 15.7%，女92.1%，三等获救比例：男13.5%，女50.0%
# # 各等级船舱中男性乘客多于女性乘客，但是女性乘客的获救比例都高于男性乘客。不同等级船舱的女性乘客的获救率高于男性，这可能是女士优先的原因

# 年龄对获救率的影响

# 按年龄分箱，将乘客分为小孩、青少年、成年人、老年人
data['Age_Lab'] = pd.cut(data['Age'], bins=[0, 12, 18, 60, 140], labels=['child', 'teenager', 'grownup', 'older'])
age_ratio = data['Survived'].groupby(data['Age_Lab']).value_counts().reset_index()
# 各年龄段总人数
age_count = age_ratio.groupby('Age_Lab')['count'].sum().reset_index()
# 合并
age_data = pd.merge(age_ratio, age_count, how='left', on=['Age_Lab'])
# 只筛选获救的数据计算获救率
age_data = age_data.query('Survived == 1')
age_data['rescue_ratio'] = age_data['count_x'] / age_data['count_y']
print(age_data)
# 0-12获救比例：57.9%；12-18获救比例：42.8%；18-60获救比例：38.8%；>60获救比例：22.7%
# 小孩、青少年、成年人、老年人的获救比例依次从高到低，小孩的获救比例最高，老年人的获救比例最低。年龄是影响获救率的重要因素

# 结论：
# - 在泰坦尼克号上，女性的获救率高于男性
# - 高等级船舱的乘客获救率高于低等级船舱
# - 小孩的获救率最高，老年人的获救率最低

# B、泰坦尼克号幸存者预测

# 数据挖掘流程
# + 数据读取与数据探索（统计分析）
#   + 读取数据与概览
#   + 单变量分析：单个变量对标签的影响
#   + 多变量分析：多个变量对标签的影响
#   + 统计绘图
# + 相关性分析（特征相关性热力图）
# + 数据清洗与预处理
#   + 缺失值处理
#   + 标准化、归一化
#   + 特征工程
# + 建模与预测
#   + 数据集切分
#   + 多模型评估优化对比：交叉验证、网格搜索
#   + 集成学习：提升算法(XGBoost、GBDT、Light-GBM)、多种算法集成(神经网络)

# 相关性分析
# 相关性分析是指对两个或多个具备相关性的变量进行分析，从而衡量变量的相关密切程度（线性相关性）
# 热力图（Heatmap）是一种数据相关性可视化工具，它通过使用颜色的深浅来展示数据矩阵中数值的大小或密度。通常来说，颜色越深，相关度越高
# 可以使用Pandas中的corr()函数来计算两个变量之间的相关系数。两个变量之间相关系数的计算公式为
#
# 即两个变量X、Y的Pearson相关系数$ρ_{X,Y}$等于它们之间的协方差$\cov(X,Y)$除以它们各自的标准差的乘积$σ_Xσ_Y$
# 值得注意的是，该相关系数只能度量变量之间的线性相关性；也就是说，相关系数越高，则变量间的线性相关程度越高
# 常见的相关系数有：
# + 皮尔森（Pearson）相关系数：最常用，适用于数据呈正态分布的场景，正态图可查看正态性，散点图展示数据关系
# + 斯皮尔曼（Spearman）相关系数：适用于数据不满足正态分布的场景，正态图可查看正态性，散点图展示数据关系
# + 肯德尔（Kendall）相关系数：适用于数据的一致性判断，通常用于非关系研究，如评委打分，数据排名等
# 按照相关的方向，相关性可以分为正相关和负相关。因此相关系数的取值始终在[-1,1]之间
# + -1：表示两个变量之间完全负线性相关
# + 0：表示两个变量之间不存在线性相关
# + 1：表示两个变量之间存在完全正线性相关
# 通常情况下通过以下取值范围判断变量的正相关强度：
# | 0.8-1.0 |  极强相关或完全相关|
# | 0.6-0.8 |  强相关|
# | 0.4-0.6 |  中等程度相关|
# | 0.2-0.4 |  弱相关|
# | 0.0-0.2 |  极弱相关或无相关|
# 通过绘制热力图，我们可以探究各个特征变量之间的相关性，进而探究特征之间是否存在多重共线性

# 计算特征之间的相关系数
data_corr = data[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
print(data_corr.corr().to_string())

# 绘制相关性热力图
# Seaborn是基于Matplotlib的Python数据可视化库。Seaborn在Matplotlib基础上进行了更高级的封装，它提供了一个高级界面，使得作图更加方便快捷
# 以下是Seaborn绘制相关性热力图的API：
# sns.heatmap()常用参数
# - annot：是否在热力图的每个单元上显示值
# - fmt：指定单元格中数据的显示格式，例如保留2位小数：".2f"
# - center：指定颜色中心值，通过该参数可以调整热力图的颜色深浅
# - linewidths ：指定每个单元格的边框宽度
# - cmap：指定热力图的填充颜色，例如summer(黄到绿)、summer_r(绿到黄)、GnBu(绿到蓝)、OrRd(橘到红)、Blues、RdBu_r

import matplotlib.pyplot as plt
import seaborn as sns

# 绘制特征相关性热力图
# 设置画面大小，dpi：图像分辨率，指定150纵轴标签水平显示
plt.figure(figsize=(8, 6), dpi=150)
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# sns.heatmap()：绘制相关性热力图
# corr()：计算属性间的相关性，不包括缺失值。默认相关性分析系数为pearson
sns.heatmap(data_corr.corr(), annot=True, linewidths=0.2, cmap="GnBu")
plt.title('特征相关性热力图')
plt.xticks(rotation=45)
# 指定分辨率保存绘图
# plt.savefig('heatmap.jpg', dpi=600)
plt.show()

# 热力图右侧的刻度展示了不同相关系数对应的颜色深浅。从图中可以看出，Parch和SibSp之间的相关性较高，为0.41，相对存在较强的多重共线性。如果相关性很强，在进行特征工程时可以考虑剔除二者中的一个变量，以免导致因多重共线性造成的过拟合

# 数据预处理
# 1）缺失值处理
# 数据集的缺失情况
# print(data.isnull().sum())
# 训练集：Age(177)、Cabin(687)、Embarked(2)
# 测试集：Age(86)、Fare(1)、Cabin(327)
# 对于缺失值的处理，我们一般选择填充和删除：
# 年龄（Age）=> 均值填充
data['Age'].fillna(data['Age'].mean(), inplace=True)
# 船票价格（Fare）=> 均值填充
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
# 座位号（Cabin）=> 缺失较多，删除
data.drop(columns='Cabin', inplace=True)
# 登船码头（Embarked）=> 众数填充
data['Embarked'].fillna(data['Embarked'].mode(), inplace=True)

# 2）特征编码
# 数值型数据我们可以直接使用；对于日期型数据，我们需要转换成单独的年月日；对于分类型数据，需要使用特征编码转换为数值
# 分类特征：Sex(male/female)、Embarked(C/Q/S)
# 编码方案如下：
# 性别（Sex）=> 男(male)：1，女(female)：0
sex_map = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_map)
# 登船码头（Embarked）=> 独热编码
embarked_dum = pd.get_dummies(data['Embarked'], prefix='Embarked', dtype=int)
# 删除源数据中的Embarked列，添加编码后的Embarked
data.drop(columns='Embarked', axis=1, inplace=True)
data = pd.concat([data, embarked_dum], axis=1)

# 非分类特征：Fare、Age
# 对于非分类特征，我们一般进行分箱处理：
# 根据样本分位数进行分箱，等比例分箱
# 船票价格(Fare) => 分箱并序数编码
data['FareBand'] = pd.qcut(data['Fare'], 4, labels=[0, 1, 2, 3])
# 删除Fare特征
data.drop(columns='Fare', inplace=True)

# 年龄(Age) => 分箱并序数编码
data['AgeBand'] = pd.cut(data['Age'], bins=[0, 12, 18, 60, 140], labels=[0, 1, 2, 3])
# 删除Age特征
data.drop(columns='Age', inplace=True)

# print(data.head().to_string())

# 特征选择与提取

# 1）特征提取
# 通过观察数据，我们发现乘客姓名中包含头衔，例如Mrs表示已婚女性。这些头衔可以将乘客进一步细分
# 提取姓名中的头衔
def extract_title(name: str):
    return name.split(',')[1].split('.')[0].strip()

# 添加头衔特征
data['Title'] = data['Name'].apply(extract_title)
# data['Title'] = data['Name'].str.extract("([A-Za-z]+)\\.")
# 查看头衔及数量
# print(data['Title'].value_counts().reset_index())
# 由于头衔类别较多，且部分不同写法但意思相同，需要整合
# 整合意思相同的头衔
data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev', 'Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Other', inplace=True)
data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
data['Title'].replace(['Mlle'], 'Miss', inplace=True)
# print(data['Title'].value_counts().reset_index())

# 头衔特征编码：序数编码
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}
data['Title'] = data['Title'].map(title_map)

# 删除Name特征
data.drop(columns='Name', axis=1, inplace=True)
# print(data.head().to_string())

# 从家庭成员变量中衍生家庭规模特征
# 通过观察数据，我们发现我们可以通过乘客兄弟姐妹及配偶人数和乘客父母及子女人数计算得到本次出行的乘客家庭规模
# 家庭规模(FamilySize) = 兄弟姐妹及配偶人数(SibSp) + 父母及子女人数(Parch) + 乘客自己(1)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# 对家庭规模特征进行分箱：1人(Alone)、2-4人(Small)、>4人(Large)
data['FamilySize'] = pd.cut(data['FamilySize'], bins=[1, 2, 5, 12], labels=['A', 'S', 'L'], right=False, include_lowest=True)
# 家庭规模特征编码：序数编码
fs_map = {'A': 0, 'S': 1, 'L': 2}
data['FamilySize'] = data['FamilySize'].map(fs_map)
# 删除SibSp、Parch特征
data.drop(columns=['SibSp', 'Parch'], inplace=True)

# 2）特征选择
# 更多的数据优于更好的算法，而更好的数据优于更多的数据。删除无关特征，最大程度保留数据
# 删除其他无关特征
data.drop(columns=['PassengerId', 'Ticket'], inplace=True)
# 应用了特征工程的数据
# print(data.head().to_string())
# 保存特征工程处理后的数据（训练集和测试集）
# data.to_csv("new_train.csv", index=False, encoding='utf-8')


# 训练集和测试集
train = pd.read_csv("new_train.csv")
# X_test = pd.read_csv("new_test.csv")
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

from sklearn.model_selection import train_test_split

# 重新划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# 1）逻辑回归（幸存者预测）

from sklearn.linear_model import LogisticRegression

# 逻辑回归分类器(二分类)（默认求解器lbfgs、分类方式OvR）
lr = LogisticRegression()
# 训练模型
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)
# print(y_pred)

# 准确度评分
print(lr.score(X_test, y_test))   # 0.7821229050279329


# 2）K近邻分类（幸存者预测）
from sklearn.neighbors import KNeighborsClassifier

# KNN分类器（默认使用标准欧几里德度量标准）
knn_clf = KNeighborsClassifier(n_neighbors=2)
# 训练模型
knn_clf.fit(X_train, y_train)

# 预测
y_pred = knn_clf.predict(X_test)
# print(y_pred)

# 平均准确度
print(knn_clf.score(X_test, y_test))   # 0.8156424581005587



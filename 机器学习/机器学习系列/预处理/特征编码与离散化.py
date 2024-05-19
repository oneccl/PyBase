#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# 数据预处理之特征编码
# https://zhuanlan.zhihu.com/p/119093636
# https://zhuanlan.zhihu.com/p/643196576
# https://runebook.dev/zh/docs/scikit_learn/modules/generated/sklearn.preprocessing.ordinalencoder

# 在机器学习中，处理离散属性（分类特征/类别特征）是一个重要的任务，需要将离散属性转换为可供模型使用的数值表示
# 机器学习算法本质上都是在基于矩阵做线性代数计算，因此参加计算的特征必须是数值型的，对于非数值型的特征需要进行编码处理
# 分类特征是用来表示分类的，分类特征是离散的，非连续的。例如性别（男/女）、等级（优/良/合格）等
# 有些分类特征也是数值，例如，账号ID、IP地址等，但是这些数值并不是连续的。连续的数字是数值特征，离散的数字是分类特征
# 对于离散型数据的编码，针对小型分类和大型分类，我们通常有以下几种常用的方式来实现，它们各有优缺点

# 小型分类特征的编码方式
# 1、序列编码（Ordinal Encoding）
# 将离散特征的各个类别映射为自然数序号，适用于类别间本来就有一定的排序关系，并且不同样本之间的距离计算有一定的意义。例如，学历中的学士(0)、硕士(1)、博士(2)，学士与硕士的距离和硕士与博士的距离相等
# 以下是序列编码的实现方式：
# 1）使用Pandas
import pandas as pd

# degree_list = ["硕士", "博士", "学士", "硕士"]
# data = pd.DataFrame(degree_list, columns=["学历"])
# # 手动编码
# ordinal_map = {"学士": 0, "硕士": 1, "博士": 2}
# data["Code"] = data["学历"].map(ordinal_map)
# print(data)
# '''
#    学历  Code
# 0  硕士     1
# 1  博士     2
# 2  学士     0
# 3  硕士     1
# '''
#
# # 2）使用Scikit-Learn库
# # Scikit-Learn库提供了序列编码API：OrdinalEncoder
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
# # https://scikit-learn.org.cn/view/744.html
# import numpy as np
# from sklearn.preprocessing import OrdinalEncoder
#
# # 序列编码器
# enc = OrdinalEncoder()
#
# # fit_transform()：拟合数据，自动编码，需要一个2D数组
# data["Code"] = enc.fit_transform(np.array(data["学历"]).reshape(-1, 1))
# print(data)
# '''
#    学历  Code
# 0  硕士   2.0
# 1  博士   0.0
# 2  学士   1.0
# 3  硕士   2.0
# '''
# # 解码
# print(enc.inverse_transform(np.array(data["Code"]).reshape(-1, 1)))
# '''
# [['硕士']
#  ['博士']
#  ['学士']
#  ['硕士']]
# '''
#
# # 3）使用Category_Encoders库
# # Category_Encoders库是一个Python第三方库，涵盖多种编码方式，且与Scikit-Learn完全兼容
# # https://contrib.scikit-learn.org/category_encoders/
# # 安装：pip install category_encoders
# import category_encoders as ce
#
# # 序列编码器
# enc = ce.OrdinalEncoder()
# # 编码
# data["Code"] = enc.fit_transform(data["学历"])
# print(data)
# '''
#    学历  Code
# 0  硕士     1
# 1  博士     2
# 2  学士     3
# 3  硕士     1
# '''
# # 解码报错：方法内部调用了get_loc(key)，key为被编码的列名，而我们这里重新赋值了，被编码的列值没变
# print(enc.inverse_transform(data["Code"]))
#
# # 2、独热编码（One Hot Encoding）
# # 独热编码使用N位状态寄存器来对N个状态（分类）进行编码，每个状态都它独立的寄存器位，并且在任意时刻只有其中一位有效
# # 例如，颜色特征有三种：红、绿、蓝，转换为独热编码分别表示为：001、010、100
# # 像颜色、品牌等这些特征就不适合使用序列编码，因为这些特征是没有排序关系的。使用独热编码可以让不同的分类处在平等的地位
# # 以下是独热编码的实现方式：
# # 1）使用Pandas
#
# colors = ['Green', 'Blue', 'Red', 'Blue']
# data = pd.DataFrame(colors, columns=['Color'])
# # pd.get_dummies()：对DataFrame指定列进行独热编码，默认返回bool类型，可通过dtype指定
# data_dum = pd.get_dummies(data['Color'], dtype=int)
# print(data_dum)
# '''
#    Blue  Green  Red
# 0     0      1    0
# 1     1      0    0
# 2     0      0    1
# 3     1      0    0
# '''
#
# # 2）使用Scikit-Learn库
# # A、LabelBinarizer：仅支持一维输入
# from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
#
# # 独热编码器
# lb = LabelBinarizer()
# # 编码
# colors_code = lb.fit_transform(colors)
# print(colors_code)
# '''
# [[0 1 0]
#  [1 0 0]
#  [0 0 1]
#  [1 0 0]]
# '''
# # 解码
# print(lb.inverse_transform(colors_code))
# '''
# ['Green' 'Blue' 'Red' 'Blue']
# '''
#
# # B、OneHotEncoder：仅支持二维输入
# # 独热编码器
# enc = OneHotEncoder()
# # 编码
# colors_code = enc.fit_transform(np.array(colors).reshape(-1, 1))
# print(colors_code.toarray())
# '''
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# '''
# # 解码
# print(enc.inverse_transform(colors_code.toarray()))
# '''
# [['Green']
#  ['Blue']
#  ['Red']
#  ['Blue']]
# '''
#
# # 3、标签编码（Label Encoding）
# # 标签编码将每个分类映射到整数值，从0开始递增。和序列编码类似，标签编码同样适用于有序特征，并且不同样本之间的距离计算要有一定的意义
# # 以下是标签编码的实现方式：
# # 1）使用Pandas
# # 标签编码：返回二元组，元素1为编码（数组类型）；元素2为分类，无重复（Index类型）
# codes, ctgs = pd.factorize(data['Color'])
# print(codes)   # [0 1 2 1]
#
# # 2）使用Scikit-Learn库
# from sklearn.preprocessing import LabelEncoder
#
# # 标签编码器
# le = LabelEncoder()
# # 编码
# colors_code = le.fit_transform(colors)
# print(colors_code)   # [1 0 2 0]
# # 解码
# print(le.inverse_transform(colors_code))
# '''
# ['Green' 'Blue' 'Red' 'Blue']
# '''
#
# # 4、频数编码（Count Encoding）
# # 频数编码将每个类别替换为该类别在数据集中的频数或出现次数。例如类别A在训练集中出现了10次，则类别A的编码为10
# # 以下是频数编码的实现方式：
# # 1）使用Pandas
# data = pd.DataFrame({
#     "City": ['beijing', 'beijing', 'beijing', 'shanghai', 'shanghai'],
#     "Val": [9, 8, 10, 9, 7]
# })
# # 使用分组计数实现
# data_count = data.groupby('City', as_index=False).agg('count')
# print(data_count)
# '''
#        City  Val
# 0   beijing    3
# 1  shanghai    2
# '''
#
# # 2）使用Category_Encoders库
# import category_encoders as ce
#
# # 频数编码编码器
# cen = ce.CountEncoder()
# # 编码
# data['Code'] = cen.fit_transform(data['City'])
# print(data)
# '''
#        City  Val  Code
# 0   beijing    9     3
# 1   beijing    8     3
# 2   beijing   10     3
# 3  shanghai    9     2
# 4  shanghai    7     2
# '''
#
# # 大型分类特征的编码方式
# # 5、目标编码（Target Encoding）
# # 目标编码是表示分类列的一种非常有效的方法，并且仅占用一个特征空间，也称为均值编码。该列中的每个值都被该类别的平均目标值替代。这可以更直接地表示分类变量和目标变量之间的关系
# # 以下是目标编码的实现方式：
# # 1）使用Pandas
# # 使用分组求和/计数实现
# data_code = data.groupby('City')['Val'].sum() / data.groupby('City')['City'].count()
# print(data_code.reset_index().rename(columns={0: 'Code'}))
# '''
#        City  Code
# 0   beijing   9.0
# 1  shanghai   8.0
# '''
#
# # 2）使用Scikit-Learn库（K折目标编码+正则化：平滑参数smooth）
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
# from sklearn.preprocessing import TargetEncoder
#
# # 目标编码器，默认5折
# ten = TargetEncoder(target_type='continuous')
# # 编码
# ten.fit_transform(np.array(data['City']).reshape(-1, 1), data['Val'])
# # 类别编码
# print(ten.encodings_)     # [array([8.92957746, 8.19480519])]
# # 类别
# print(ten.categories_)    # [array(['beijing', 'shanghai'], dtype=object)]
# # 目标的总体平均值
# print(ten.target_mean_)   # 8.6
#
# # 3）使用Category_Encoders库（正则化：平滑参数smoothing）
# # https://contrib.scikit-learn.org/category_encoders/targetencoder.html
# from category_encoders.target_encoder import TargetEncoder
#
# # 目标编码器
# ten = TargetEncoder(cols=['City'])
#
# # 编码
# data['Code'] = ten.fit_transform(data['City'], data['Val'])
# print(data)
# '''
#        City  Val      Code
# 0   beijing    9  8.661786
# 1   beijing    8  8.661786
# 2   beijing   10  8.661786
# 3  shanghai    9  8.514889
# 4  shanghai    7  8.514889
# '''

# 6、哈希编码（Hashing Encoding）
# 哈希编码将哈希函数应用于变量，将任意数量的变量以一定的规则映射到给定数量的变量。特征哈希可能会导致元素之间发生冲突
# 以下是哈希编码的实现方式：
# https://contrib.scikit-learn.org/category_encoders/hashing.html

# from category_encoders.hashing import HashingEncoder
#
# # 哈希编码器
# hen = HashingEncoder(cols=['City'], n_components=4)
#
# # 编码（此处报错，原因未知，后续补充解决）
# print(hen.fit_transform(data['City'], data['Val']))

# 7、离散化
# 离散化（也称量化或分箱）是一种数据预处理技术，用于将连续的数值型的数据转换为离散的分类的标签。某些具有连续特征的数据集可能会从离散化中受益，因为离散化可以将连续属性的数据集转换为仅具有名义属性的数据集
# 这种处理方式主要应用于一些需要转化为分类问题的数据集，如机器学习和数据挖掘中的输入变量。离散化的原理主要是通过将连续的数值属性转化为离散的数值属性来实现数据的转化
# 这个过程通常会采用分箱（Binning）的方法。在分箱中，原数据的值被分配到一些离散的、预定义的类别中，这些类别通常被称为”箱子”或“桶”，箱子的数量和大小可以根据数据的分布和实际需求进行调整
# 常用的离散化处理方式有两种：特征二值化和K-bins离散化

# 1）特征二值化
# 特征二值化是将数字特征用阈值过滤以获得布尔值的过程。这对于下游概率估计器很有用，这些估计器假设输入数据是根据多元Bernoulli（伯努利）分布进行分布的
# 特征二值化的原理是根据阈值将一系列连续的数值分为两种类别。二值化的应用场景有：垃圾邮件的判定，信用卡欺诈的判定，医疗检测结果（阴性阳性）的判定等

# # Scikit-Learn提供了数据二值化处理的API：Binarizer
# # https://scikit-learn.org.cn/view/721.html
# # 以下是一个示例：
# from sklearn.preprocessing import Binarizer
# import numpy as np
#
# data = np.random.randint(0, 10, (3, 3))
# print(data)
# '''原数据
# [[7 3 1]
#  [6 9 7]
#  [9 3 9]]
# '''
# # threshold参数用于控制二值化分类阈值
# binarizer = Binarizer(threshold=7)
# # 二值化转换
# print(binarizer.fit_transform(data))
# '''二值化后的数据
# [[0 0 0]
#  [0 1 0]
#  [1 0 1]]
# '''
# # 上述示例中，我们设置二值化分类的阈值`threshold`为7，表示大于7的类别为1，小于等于7的类别为0
#
# # 2）K-bins离散化
# # K-bins离散化将特征离散到K个容器中。K-bins离散化的原理则是将连续的数值分成K个类别
# # K-bins离散化的应用有：根据用户的购买行为将用户划分为不同的消费类别等。这些场景下，不能简单的进行二值化，需要离散化为多个分类
#
# # Scikit-Learn提供了数据K-bins离散化的API：KBinsDiscretizer
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
# # https://scikit-learn.org.cn/view/722.html
# # API常用参数及说明如下：
# # | 参数 | 说明 |
# # | `n_bins`  | 产生的箱数（分类数），默认值为5。如果小于2，则引发ValueError
# # | `encode`  | 对转换结果进行编码的方法，默认为`onehot`，使用独热编码，返回一个稀疏矩阵；其他参数有：`onehot-dense`使用独热编码，返回一个密集数组；`ordinal`使用序列编码
# # | `strategy`| 定义箱子宽度的策略，默认为`quantile`，每个特征中的所有bin具有相同的点数；其他参数有：`uniform`每个特征中的所有bin都具有相同的宽度；`kmeans`每个bin中的值都具有相同一维K均值簇最近中心
# # | `dtype`  | 输出数据的类型，默认为None，表示与输入类型一致。其他仅支持np.float32和np.float64
# # | `subsample` | 用于拟合模型的最大样本数，默认为`warn`(200000)。显式设置为None可以消除此警告
# # 常用属性及说明如下：
# # | 属性 | 说明|
# # | `bin_edges_`| 箱子的边缘，每个特征的边缘为数组类型，数组的首末元素为原数据的最小值与最大值
# # | `n_bins_` | 箱数（分类数），宽度`<=1e-8`的箱子将被移除，并发出警告
# # 以下是一个示例：
# from sklearn.preprocessing import KBinsDiscretizer
#
# data = np.random.randint(0, 100, (10, 1))
# print(data)
# '''
# [[16]
#  [ 5]
#  [ 3]
#  [18]
#  [74]
#  [20]
#  [22]
#  [54]
#  [89]
#  [65]]
# '''
# # 使用序列编码，每个箱子（分类）具有相同的宽度
# est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform', subsample=None)
# # K-bins离散化转换
# data_trans = est.fit_transform(data)
# print(data_trans)
# '''
# [[0.]
#  [0.]
#  [0.]
#  [0.]
#  [3.]
#  [0.]
#  [0.]
#  [2.]
#  [3.]
#  [2.]]
# '''
# # 箱数（分类数）
# print(est.n_bins_)     # [4]
# # 每个特征箱子的边缘
# print(est.bin_edges_)  # [array([ 3. , 24.5, 46. , 67.5, 89. ])]
#
# # 将分箱数据转换到原始特征空间。每个值将等于两个bin边缘的平均值
# print(est.inverse_transform(data_trans))
# '''
# [[13.75]
#  [13.75]
#  [13.75]
#  [13.75]
#  [78.25]
#  [13.75]
#  [13.75]
#  [56.75]
#  [78.25]
#  [56.75]]
# '''
# # 上述示例中，我们设置分类数为4，`encode`参数设置了使用序列数进行编码，`strategy`参数设置了将原始数据进行均分


# Pandas分组/分箱、离散化
# Pandas提供了智能剪贴功能：`pd.cut()`与`pd.qcut()`，它们通常用于更方便直观地处理关系型或标签型数据，将数据进行分箱/离散化

# pd.cut()
# 我们可以通过2种方式使用`cut()`函数：直接指定`bin`的数量，让Pandas为我们计算等大小的`bin`，或者我们可以按照自己的意愿手动指定`bin`的边缘
# 在`cut()`函数中，`bin`边缘的间距大小是相等的，每个`bin`或箱中的元素数量不均匀
# 例如，如果对年龄进行分箱，0-1岁是婴儿，1-12岁是孩子，12-18岁是青少年，18-60岁是成年人，60岁以上是老年人。所以我们可以设置bins=[0, 1, 12, 18, 60, 140]和labels=['infant', 'kid', 'teenager', 'grownup', 'senior citizen']

# pd.cut(x,bins,right,labels,retbins,precision,include_lowest,duplicates,ordered)
# - x：需要进行分箱的数据，1D数组或系列类型，如果数据存在`NaN`则报错
# - bins：分箱的边界，如果是单个整数，则表示基于数据中的最小值和最大值生成等间距间隔；也可以是自定义边界值的列表或数组
# - right：是否包含最右边的数值，默认True（右闭）
# - labels：分箱的标签，长度保持与分箱数一致
# - retbins：是否显示分箱的边界值，默认为False。当`bins`为整数时设置True可以显示边界值
# - precision：分箱边界的精度，默认3位小数
# - include_lowest：是否包含最左边的数值，默认False（左开）
# - duplicates：默认为`raise`，如果分箱的边界不唯一，则引发ValueError；指定`drop`则去重
# - ordered：标签是否有序，默认True，分类结果将被排序

# 以下是一个使用示例：

# # 数据准备
# years = [2024, 2023, 2017, 2011, 2015, 2023, 2008, 2010]
# df = pd.DataFrame(years, columns=['Year'])
# # 左开右闭
# print(pd.cut(df['Year'], bins=3, precision=0))
# 数据的年份范围是2008年到2024年（16个），当我们指定`bins=3`时，Pandas将它切分成3个等宽的`bin`，每个`bin`5-6年
# 需要注意的是，Pandas会自动将第一类的下限值丢失精度，以确保将2008年也包括在结果中
# # 左闭右闭
# print(pd.cut(df['Year'], bins=3, include_lowest=True, precision=0))
# 可以将标签参数指定为一个列表，而不是获得间隔，以便更好地分析
# 指定分类标签
# df['Label'] = pd.cut(df['Year'], bins=3, labels=['Old', 'Medium', 'New'], precision=0)
# print(df)
# # 如果指定`labels=False`，我们将得到`bin`的数字表示（从0依次开始递增）
# df['Label'] = pd.cut(df['Year'], bins=3, labels=False)
# print(df)
#
# # 查看每个分箱中的值数量
# print(df['Label'].value_counts().reset_index())

# # 显示分箱的边界值
# # 我们可以指定`retbins=True`一次性获得bin区间和边界值离散的序列，此时方法返回一个二元组
# cut_series, cut_intervals = pd.cut(df['Year'], bins=3, retbins=True, precision=0)
# # bin区间
# print(cut_series)
# # 分箱的边界值
# print(cut_intervals)

# 我们也可以通过给`bins`参数传入一个列表来手动指定`bin`的边缘
# 自定义bin边缘
# print(pd.cut(df['Year'], bins=[2008, 2010, 2020, 2024], include_lowest=True))
# 这里，我们设置了`include_lowest=True`，默认情况下，它被设置为False，因此，当Pandas看到我们传递的列表时，它将把2008年排除在计算之外。因此，这里我们也可以使用任何小于2008的值

# pd.qcut()
# `qcut()`函数（Quantile-Cut）与`cut()`的关键区别在于，在`qcut()`中，每个`bin`中的元素数量将大致相同，但这将以不同大小的`bin`区间宽度为代价
# 在`qcut()`函数中，当我们指定`q=5`时，我们告诉Pandas将数据列切成5个相等的量级，即0-20%，20-40%，40-60%，60-80%和80-100%桶/箱

# pd.qcut(x,q,labels,retbins,precision,duplicates)
# - x：需要进行分箱的数据，1D数组或系列类型，如果数据存在NaN则保留
# - q：分位数，例如4用于四分位，也可以指定为列表[0,0.25,0.5,0.75,1]
# 其他参数使用同pd.cut()

# # 添加NaN值
# df.loc[0, 'Year'] = np.NaN
#
# # 左开右闭
# print(pd.qcut(df['Year'], q=3))
# # 你是否注意到，在输出结果中，NaN值也被保留为NaN？
#
# # 指定分箱标签
# df['Label'] = pd.qcut(df['Year'], q=3, labels=['Old', 'Medium', 'New'])
# print(df)
#
# # 自定义分箱量级
# print(pd.qcut(df['Year'], q=[0, 1 / 3, 2 / 3, 1]))
# # 可以看到，分箱的边缘是不等宽的，因为它要容纳每个桶1/3的值，因此它要自己计算每个箱子的宽度来实现这一目标




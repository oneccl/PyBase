
# pandas相关函数示例

import numpy as np
import pandas as pd

'''
1、concat()拼接
'''
df1 = pd.DataFrame({'name': ['a', 'b', 'b'], 'age': [18, 20, 19]})
df2 = pd.DataFrame({'name': ['m', 'n'], 'age': [21, 22], 'addr': ['CN', 'US']})
# print(df1)
# print(df2)

# pd.concat(objs,axis=0,join='outer'): df拼接
# objs：需要连接的对象，如[df1,df2]，需要使用[]包裹
# axis=0（默认）：要连接的轴，上下/纵向拼接，没有的使用NaN填充；axis=1：左右/水平拼接，没有的使用NaN填充
# join='outer'（默认）：外连接，会保留两个表的全部信息；join='inner'：内连接：只保留两个表的公共信息；如何处理其他轴上的索引，outer为并集，inner为交集
# ignore_index：默认False，索引可能重复，若设置True，则重新标记0~n-1
# keys：用于构建最外层分层索引

df_concat = pd.concat([df1, df2])
# print(df_concat)
df_concat = pd.concat([df1, df2], axis=1)
# print(df_concat)

df1 = pd.DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]})
df2 = pd.DataFrame({'A': [2, 3, 5], 'C': [2, 5, 6]})
print(df1.to_string())
'''
   A  B
0  1  2
1  3  4
2  5  6
'''
print(df2.to_string())
'''
   A  C
0  2  2
1  3  5
2  5  6
'''
# 默认join=outer并集
print(pd.concat([df1, df2], ignore_index=True))
'''
   A    B    C
0  1  2.0  NaN
1  3  4.0  NaN
2  5  6.0  NaN
3  2  NaN  2.0
4  3  NaN  5.0
5  5  NaN  6.0
'''
# 交集
print(pd.concat([df1, df2], ignore_index=True, join='inner'))
'''
   A
0  1
1  3
2  5
3  2
4  3
5  5
'''
# 添加多级索引
print(pd.concat([df1, df2], axis=1, keys=['t1', 't2']))
print(pd.concat({'t1': df1, 't2': df2}, axis=1))
'''
  t1    t2   
   A  B  A  C
0  1  2  2  2
1  3  4  3  5
2  5  6  5  6
'''
# 将行追加到数据帧
# 将df2的第2行追加到df1后面
print(pd.concat([df1, df2.loc[2:2, :]], ignore_index=True))
'''
   A    B    C
0  1  2.0  NaN
1  3  4.0  NaN
2  5  6.0  NaN
3  5  NaN  6.0
'''

'''
2、merge()连接
'''
df1 = pd.DataFrame({'name': ['a', 'b', 'c'], 'id_card': [1, 3, 5]})
df2 = pd.DataFrame({'id_card': [1, 2, 3, 4, 5], 'birth': [21, 22, 23, 24, 25], 'addr': ['CN', 'US', 'CN', 'US', 'CN']})
# print(df1)
# print(df2)

# pd.merge(left,right,how='inner',on=None,left_on=None,right_on=None): df连接
# left：左表  right：右表
# how='inner'(默认)：连接方式：inner内连接：两表公共；outer外连接：左连接与右连接的并集；left左连接：左表全部，右表匹配；right右连接：右表全部，左表匹配
# on=None(默认)：连接条件(连接列名)，必须同时存在于两个表中；若未指定，则以left和right列名的交集作为连接条件，可指定多个连接键
# left_on,right_on：当两边字段名不同时，可以使用left_on,right_on设置连接
# sort：默认True，默认按连接键的字典顺序对结果数据帧进行排序

pd_merge = pd.merge(df1, df2, how='left', on='id_card')   # 左连接
# print(pd_merge)
pd_merge = pd.merge(df1, df2, how='right', on='id_card')  # 右连接
# print(pd_merge)
pd_merge = pd.merge(df1, df2, how='inner', on='id_card')  # 内连接
# print(pd_merge)
pd_merge = pd.merge(df1, df2, how='outer', on='id_card')  # 外连接
# print(pd_merge)
# 连接字段名不同时
df1 = pd.DataFrame({'name': ['a', 'b', 'c'], 'cla_id': [1, 2, 1]})
df2 = pd.DataFrame({'id': [1, 2, 3], 'cla_name': ['cla1', 'cla2', 'cla3']})
# print(df1.to_string())
# print(df2.to_string())
pd_merge = pd.merge(df1, df2, how='left', left_on='cla_id', right_on='id').drop(columns='id')
# print(pd_merge)

df1 = pd.DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]})
df2 = pd.DataFrame({'A': [2, 3, 5], 'C': [2, 5, 6]})
print(df1.merge(df2, on='A', how='outer'))
'''
   A    B    C
0  1  2.0  NaN
1  3  4.0  5.0
2  5  6.0  6.0
3  2  NaN  2.0
'''
# 重复列会自动重命名
df3 = pd.DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6], 'C': [6, 8, 9]})
df4 = pd.DataFrame({'A': [2, 3, 5], 'C': [2, 5, 6]})
print(pd.merge(df3, df4, on='A', how='outer'))
'''
   A    B  C_x  C_y
0  1  2.0  6.0  NaN
1  3  4.0  8.0  5.0
2  5  6.0  9.0  6.0
3  2  NaN  NaN  2.0
'''

'''
3、transform()变换函数：转换聚合
# 支持多函数：生成多层索引DataFrame，第一层是原始数据集的列名；第二层是transform()调用的函数名
'''
df = pd.DataFrame({'name': ['a', 'b', 'c', 'd', 'e'], 'cla_id': [1, 2, 2, 1, 1], 'score': [80, 70, 80, 100, 90]})
# print(df)

# 1）transform()作用于Series
series = df['score'].transform(lambda s: float(s))
# print(series)
# 2）transform()作用于DataFrame：会将传入的函数作用于每一列
df_d = df.loc[:, 'cla_id': 'score'].transform([np.abs, lambda x: x-1])
print(df_d)
'''
    cla_id             score         
  absolute <lambda> absolute <lambda>
0        1        0       80       79
1        2        1       70       69
2        2        1       80       79
3        1        0      100       99
4        1        0       90       89
'''
df1 = df_d.stack(level=-1)
print(df1)
'''
            cla_id  score
0 absolute       1     80
  <lambda>       0     79
1 absolute       2     70
  <lambda>       1     69
2 absolute       2     80
  <lambda>       1     79
3 absolute       1    100
  <lambda>       0     99
4 absolute       1     90
  <lambda>       0     89
'''
# 3）transform()作用于groupby分组后：保留原数据，计算统计值添加进去：eg: 在原数据集中增加一列：每个班平均分avg_score
df['avg_score'] = df.groupby('cla_id')['score'].transform('mean')
# print(df)
df['各班人数'] = df.groupby('cla_id')['name'].transform('count')
# print(df)
df['各班总成绩'] = df.groupby('cla_id')['score'].transform('sum')
# print(df)

'''
4、sort_values()按值排序
'''
# sort_values(by,axis=0,ascending=True,inplace=False) 按值排序
# by：需要排序的行或列
# axis：按行排序（默认0）还是按列排序（1）：按行比的是列的每行
# ascending=True：升序（默认True）还是降序（False）
# na_position：指定如何处理缺失值（NaN）
#   first：将缺失值放在排名结果的顶部
#   last：默认值，将缺失值放在排名结果的底部
new_df = df.sort_values(by='score', axis=0, ascending=False)
# print(new_df)

df = pd.DataFrame({
    'A': pd.Timestamp('20230502'),
    'B': pd.Series([1, 2, 3], index=list(range(3)), dtype='float64'),
    'C': np.array([4] * 3, dtype='int64')
})
print(df.to_string())
'''
           A    B  C
0 2023-05-02  1.0  4
1 2023-05-02  2.0  4
2 2023-05-02  3.0  4
'''
df2 = df.sort_values(by='B', ascending=False)
print(df2.to_string())
'''
           A    B  C
2 2023-05-02  3.0  4
1 2023-05-02  2.0  4
0 2023-05-02  1.0  4
'''

dfx = pd.DataFrame({'A': [2, 5, 3, 7], 'B': [8, np.nan, 5, 6]})
dfx.sort_values('A', ignore_index=True, inplace=True)
print(dfx.to_string())
'''
   A    B
0  2  8.0
1  3  5.0
2  5  NaN
3  7  6.0
'''
dfx.sort_values('B', na_position='first', ignore_index=True, inplace=True)
print(dfx.to_string())
'''
   A    B
0  5  NaN
1  3  5.0
2  7  6.0
3  2  8.0
'''

# 补充：按轴排序 sort_index(axis, ascending)
df1 = df.sort_index(axis=1, ascending=False)
print(df1.to_string())
'''
   C    B          A
0  4  1.0 2023-05-02
1  4  2.0 2023-05-02
2  4  3.0 2023-05-02
'''

'''
5、rank()计算排名
'''
# rank(axis=0,method='average',ascending=True) 计算排名：保留原数据，计算统计值添加进去
# axis：按行排序（默认0）还是按列排序（1）：按行比的是列的每行
# method：
#   average：2个值相同时都取平均排名，如 1 2.5 2.5 4 ...
#   first：2个值相同按位置排序，谁在前排名靠前
#   min：2个值相同时都取较小的排名，下一个为较小的排名+2，如 1 2 2 4 ...
#   max：2个值相同时都取较大的排名，下一个为较大的排名+1，如 1 3 3 4 ...
#   dense：2个值相同时，类似row_number()排序，如 1 2 3 4 ...
# ascending=True：升序（默认True）还是降序（False）
# pct：返回相对排名（每个值在数据中的位置的百分比），百分比表示每个元素在数据集中的相对位置，默认False
# na_option：指定如何处理缺失值（NaN）
#   keep：默认值，缺失值不参与排名
#   top：将缺失值放在排名结果的顶部
#   bottom：将缺失值放在排名结果的底部
df['average_rk'] = df['score'].rank(method='average', ascending=False)
df['first_rk'] = df['score'].rank(method='first', ascending=False)
df['min_rk'] = df['score'].rank(method='min', ascending=False)
df['max_rk'] = df['score'].rank(method='max', ascending=False)
df['dense_rk'] = df['score'].rank(method='dense', ascending=False)
# print(df.to_string())

# 其它使用见：分析案例/多列排序与排名.py

'''
6、apply(lambda/func)：行列级应用函数
'''
# apply(lambda)：Python lambda表达式只能包含一条语句；保留原数据，计算统计值添加进去
total_score = sum(df['score'].values)
# print(total_score)      # 计算某列总和
df['分数占比'] = df['score'].apply(lambda x: x/total_score)
# print(df.to_string())   # 计算占比

df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('5/1/2023', periods=10))
print(df.to_string())
# 提取每列的最大值和最小值对应的日期
# Series与DataFrame的idxmax()与idxmin()函数用于计算最大值与最小值对应的索引
df_max = df.apply(lambda x: x.idxmax()).reset_index()
print(df_max)
'''
  index          0
0     A 2023-05-04
1     B 2023-05-04
2     C 2023-05-03
'''
df_min = df.apply(lambda x: x.idxmin()).reset_index()
print(df_min)
'''
  index          0
0     A 2023-05-01
1     B 2023-05-01
2     C 2023-05-06
'''
print(df_min.set_index('index').T)
'''
index          A          B          C
0     2023-05-06 2023-05-04 2023-05-07
'''

'''
7、同比环比
'''
import datetime
# print(pd.date_range('20210601','20221201', freq='MS'))
dts = [datetime.datetime.strftime(dt, '%Y%m%d') for dt in pd.date_range('20210601', '20221201', freq='MS')]
sales = [np.random.randint(50, 100) for i in range(19)]
# print(dts)
# print(sales)
# Python计算环比同比：pct_change(periods)：periods=1环比；periods=12同比
data = {'dt': pd.date_range('20210601', '20221201', freq='MS'), 'sale': [np.random.randint(50, 100) for i in range(19)]}
df = pd.DataFrame(data)
# print(df.to_string())
# 1）计算之前值：shift(periods=1)
# 同比上月值计算（前提：已对dt排序(连续)，dt值不重复）：shift(1)：fillna(0) = replace(np.nan, 0)
df['last_m_sale'] = df['sale'].shift(1).fillna(0)
# 环比去年同期值计算（前提：已对dt排序(连续)，dt值不重复）：shift(12)：fillna(0) = replace(np.nan, 0)
df['last_y_sale'] = df['sale'].shift(12).fillna(0)
# print(df.to_string())
# 2）计算同比环比变化：diff()
# 同比变化：当前月值 - 上个月值：diff(periods=1)
df['last_m_change'] = df['sale'].diff().replace(np.nan, 0)
# 环比变化：当前月值 - 去年同期值：diff(12)
df['last_y_change'] = df['sale'].diff(12).replace(np.nan, 0)
# print(df.to_string())
# 3）计算同比环比：pct_change(periods=1)：periods=1环比；periods=12同比
# 环比计算：pct_change()
df['环比'] = round((df['sale']-df['last_m_sale']) / df['last_m_sale'], 4)   # 方法1：除数为0结果为inf
df['环比-P'] = round(df['sale'].pct_change(), 4)                            # 方法2：除数为0结果为NaN
# 同比计算：pct_change(periods=12)：periods=1（默认1）用来设置计算的周期
df['同比'] = round((df['sale']-df['last_y_sale']) / df['last_y_sale'], 4)   # 方法1：除数为0结果为inf
df['同比-P'] = round(df['sale'].pct_change(periods=12), 4)                  # 方法2：除数为0结果为NaN
print(df.to_string())

# 案例：指定列计算同比纵向拼接
df = pd.DataFrame({'Y': [2020, 2021, 2022], 'A': [10, 20, 15], 'B': [20, 25, 30]})
# print(df.to_string())
'''
      Y   A   B
0  2020  10  20
1  2021  20  25
2  2022  15  30
'''
df1 = df.set_index('Y')
df2 = df1[df1.columns].pct_change().reset_index()   # 将Y列设置的索引还原到df
# print(df2.to_string())
'''
      Y     A     B
0  2020   NaN   NaN
1  2021  1.00  0.25
2  2022 -0.25  0.20
'''
# 纵向拼接
df_concat = pd.concat([df, df2])
# print(df_concat.to_string())

'''
8、value_counts()
'''
'''
Series.value_counts(normalize,sort,ascending,bins,dropna)
DataFrame.value_counts(subset,normalize,sort,ascending,dropna)
- subset：Series对象
- normalize：是否对统计结果进行标准化，默认False，如果为True，则返回统计的相对频率（频数占比，而非频数）
- sort：是否对统计结果按频率排序，默认True，按频率排序，如果为False，则不进行排序
- ascending：升序还是降序排序，默认False，降序排序，如果为True，则升序排序
- bins：如果指定了bins参数，则将数值数据进行分箱，并计算每个箱子的频数，仅适用于数值数据
- dropna：是否只统计非缺失值NaN的频数，默认True，如果为False，则包括缺失值的频数
'''

df = pd.DataFrame({'A': ['a', 'c', 'a', np.NaN], 'B': [2, 6, 3, 7]})

print(df.value_counts(df['A'], dropna=False).reset_index())
'''
     A  count
0    a      2
1    c      1
2  NaN      1
'''
print(df['A'].value_counts(normalize=True).reset_index())
'''
   A  proportion
0  a    0.666667
1  c    0.333333
'''

# 指定字段对该字段值计数，统计结果生成的列字段名默认为count，可重命名
df = pd.DataFrame({'name': ['a', 'b', 'a'], 'age': [18, 20, 19]})
# print(df)
df1 = df['name'].value_counts().reset_index().rename(columns={'index': 'name', 'count': 'n_count'})
# print(df1)
''' pandas 2.0.3 输出结果：会将统计结果列命名为count，原来被统计的列名不变
  name  n_count
0    a          2
1    b          1
'''
''' pandas 1.4.2 输出结果：会将统计结果列命名为被统计的列名，将原来被统计的列名命名为index
   name  name
0     a     2
1     b     1
'''
# pandas 1.4.2 版本可将上面语句改为如下：这样输出结果会与pandas 2.0.3 相同
# df1 = df['name'].value_counts().reset_index().rename(columns={'index': 'name', 'name': 'n_count'})

'''
9、set_index()、reset_index()
'''
'''
set_index(['col1',...]) 将某些列值设置为行索引
reset_index() 重置索引（还原set_index()设置的索引）将原来的索引index作为新的列添加到df
'''
# set_index(str/list, drop, append, inplace)
'''
- str/list：需要设置成索引的列
- drop：设置为索引后是否删除该列（默认True）
- append：保持现有索引，追加设置索引
- inplace：是否修改原数据
'''
# reset_index(level, drop, inplace)

# 示例1：见本文7、同比环比
# 示例2：loc[row_index] 选取一行或多行（会转列显示，列字段会默认显示在索引中）
df = pd.DataFrame({'Y': [2020, 2021, 2022], 'A': [10, 20, 15], 'B': [20, 25, 30]})
# print(df.to_string())
'''
      Y   A   B
0  2020  10  20
1  2021  20  25
2  2022  15  30
'''
df_2 = df.loc[2]
# print(df_2.to_string())
'''
Y    2022
A      15
B      30
'''
df_2 = df_2.reset_index()
# print(df_2.to_string())
'''
  index     2
0     Y  2022
1     A    15
2     B    30
'''
# 重命名字段
df_2.columns = df_2.values.tolist()[0]
df_2.drop(index=0, inplace=True)
df_2.rename(columns={'Y': 'C'}, inplace=True)
# print(df_2.to_string())
'''
   C  2022
1  A    15
2  B    30
'''
# 列转行（详见本文10、行转列、列转行）
df_2 = pd.melt(df_2, id_vars='C', var_name='Y', value_name='V')
# print(df_2.to_string())
'''
   C     Y   V
0  A  2022  15
1  B  2022  30
'''
# 示例3：见本文10、行转列、列转行
# 示例4：见本文11、groupby分组

# 设置多级行索引名称
# df.index.set_names(['lvl1', 'lvl2'])    # 例如设置2级行索引


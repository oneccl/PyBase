
# fillna()填充、多级列索引数据处理
# Python中的缺失值

import numpy as np
import pandas as pd

# A、fillna()填充、多级列索引数据处理

# fillna(value,method,axis,inplace,limit)
# - value：用来填充缺失值的常数或字典，可使用字典为多列填充不同的常数
# - method：填充方式，向下/右（ffill）或向上/左（bfill）填充
# - axis：对缺失值填充的轴，默认0（按列填充），1为按行填充
# - inplace：是否修改原数据帧
# - limit：限制向前或向后填充的最大数量


# 1、数据准备
# 多级列索引（类型1）
arr = np.array([[1, 4, 1, 3], [2, 3, np.nan, 4], [2, np.nan, np.nan, 7], [3, 5, np.nan, 2], [4, 2, 3, 5]])
dfx = pd.DataFrame(arr, columns=pd.MultiIndex.from_product([['bar', 'foo'], ['one', 'two']]))
print(dfx.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  NaN  4.0
2  2.0  NaN  NaN  7.0
3  3.0  5.0  NaN  2.0
4  4.0  2.0  3.0  5.0
'''

# 2、fillna()填充
# 1）整个df填充操作
# 1.1）对整个df按列向下填充
dfx1 = dfx.fillna(method='ffill')
print(dfx1.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  1.0  4.0
2  2.0  3.0  1.0  7.0
3  3.0  5.0  1.0  2.0
4  4.0  2.0  3.0  5.0
'''
# 1.2）对整个df按列向上填充
dfx2 = dfx.fillna(method='bfill')
print(dfx2.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  3.0  4.0
2  2.0  5.0  3.0  7.0
3  3.0  5.0  3.0  2.0
4  4.0  2.0  3.0  5.0
'''
# 1.3）对整个df按行向右填充
dfx3 = dfx.fillna(method='ffill', axis=1)
print(dfx3.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  3.0  4.0
2  2.0  2.0  2.0  7.0
3  3.0  5.0  5.0  2.0
4  4.0  2.0  3.0  5.0
'''
# 1.4）对整个df按行向左填充
dfx4 = dfx.fillna(method='bfill', axis=1)
print(dfx4.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  4.0  4.0
2  2.0  7.0  7.0  7.0
3  3.0  5.0  2.0  2.0
4  4.0  2.0  3.0  5.0
'''
# 2）指定行填充操作
# 指定行向右填充
dfx.iloc[2] = dfx.iloc[2].fillna(method='ffill')
dfx.iloc[2].fillna(method='ffill', inplace=True)
print(dfx.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  NaN  4.0
2  2.0  2.0  2.0  7.0
3  3.0  5.0  NaN  2.0
4  4.0  2.0  3.0  5.0
'''
# 3）指定列填充操作
# 指定列向下填充
dfx.loc[:, ('foo', 'one')].fillna(method='ffill', inplace=True)
print(dfx.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0  1.0  4.0
2  2.0  NaN  1.0  7.0
3  3.0  5.0  1.0  2.0
4  4.0  2.0  3.0  5.0
'''
# 4）对多个列填充不同的常数
dfx5 = dfx.fillna({('bar', 'two'): 'X', ('foo', 'one'): 'Y'})
print(dfx5.to_string())
'''
   bar       foo     
   one  two  one  two
0  1.0  4.0  1.0  3.0
1  2.0  3.0    Y  4.0
2  2.0    X    Y  7.0
3  3.0  5.0    Y  2.0
4  4.0  2.0  3.0  5.0
'''

# 以上案例都会报出如下警告：
# FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
# fillna()的method参数和直接使用ffill()、bfill()效果一样，将来会被遗弃


# B、多级列索引数据处理

# 1、数据准备
# 多级列索引（类型2）
arr = np.array([[1, 4, 1, 3], [2, 3, np.nan, 4], [2, np.nan, np.nan, 7], [3, 5, np.nan, 2], [4, 2, 3, 5]])
df = pd.DataFrame(arr, columns=pd.MultiIndex.from_arrays([['bar', 'bar', 'foo', 'foo'], ['one', 'two', 'three', 'four']]))
print(df.to_string())
'''
   bar        foo     
   one  two three four
0  1.0  4.0   1.0  3.0
1  2.0  3.0   NaN  4.0
2  2.0  NaN   NaN  7.0
3  3.0  5.0   NaN  2.0
4  4.0  2.0   3.0  5.0
'''

# 2、需求：将数据转换为3列：第一层列索引为一列，第二层列索引为一列，数据项为一列，最终结果数据（NaN）不能丢失

# 1）方式1：使用中间表进行join连接

# 获取指定层级列索引
print(df.columns.get_level_values(0).tolist())
'''
['bar', 'bar', 'foo', 'foo']
'''
# 将多层列索引全部放到行中（列转行）
# 创建多级列索引间的对应关系作为中间表，fillna(method='ffill') <=> ffill() 根据前面非空值向后填充
df_mid = pd.DataFrame(df.columns.tolist()).rename(columns={0: 'a', 1: 'c'})
print(df_mid.to_string())
'''
     a      c
0  bar    one
1  bar    two
2  foo  three
3  foo   four
'''
# droplevel(level,axis)：删除指定层级索引，默认axis=0行索引
df_d = df.droplevel(level=0, axis=1)
print(df_d.to_string())
'''
   one  two  three  four
0  1.0  4.0    1.0   3.0
1  2.0  3.0    NaN   4.0
2  2.0  NaN    NaN   7.0
3  3.0  5.0    NaN   2.0
4  4.0  2.0    3.0   5.0
'''
# 所有列全部转行
df_m = pd.melt(df_d, var_name='c', value_name='v')
print(df_m.to_string())
'''
        c    v
0     one  1.0
1     one  2.0
2     one  2.0
3     one  3.0
4     one  4.0
5     two  4.0
6     two  3.0
7     two  NaN
8     two  5.0
9     two  2.0
10  three  1.0
11  three  NaN
12  three  NaN
13  three  NaN
14  three  3.0
15   four  3.0
16   four  4.0
17   four  7.0
18   four  2.0
19   four  5.0
'''
# 合并
df1 = pd.merge(df_mid, df_m, on='c')
print(df1.to_string())
'''
      a      c    v
0   bar    one  1.0
1   bar    one  2.0
2   bar    one  2.0
3   bar    one  3.0
4   bar    one  4.0
5   bar    two  4.0
6   bar    two  3.0
7   bar    two  NaN
8   bar    two  5.0
9   bar    two  2.0
10  foo  three  1.0
11  foo  three  NaN
12  foo  three  NaN
13  foo  three  NaN
14  foo  three  3.0
15  foo   four  3.0
16  foo   four  4.0
17  foo   four  7.0
18  foo   four  2.0
19  foo   four  5.0
'''

# 2）方式2：使用stack()
# 测试：将指定某一层列索引放到行中（行转列）
# 参数dropna默认为True，默认删除转换后值都为NaN的行，会导致数据丢失
df2 = df.stack(level=0, dropna=False).reset_index(level=-1)
print(df2.to_string())
'''
  level_1  one  two  three  four
0     bar  1.0  4.0    NaN   NaN
0     foo  NaN  NaN    1.0   3.0
1     bar  2.0  3.0    NaN   NaN
1     foo  NaN  NaN    NaN   4.0
2     bar  2.0  NaN    NaN   NaN
2     foo  NaN  NaN    NaN   7.0
3     bar  3.0  5.0    NaN   NaN
3     foo  NaN  NaN    NaN   2.0
4     bar  4.0  2.0    NaN   NaN
4     foo  NaN  NaN    3.0   5.0
'''
# 将多层列索引全部放到行中（列转行）
# stack()多层索引一起转行时，会将多层索引的各层索引值两两匹配，导致结果数据增多，最终导致结果错误（仅将某一层放入行时不会出现该情况）
df3 = df.stack(level=[0, -1], dropna=False).reset_index(level=[0, -1, -2]).drop(columns='level_0')
print(df3.to_string())
'''
   level_1 level_2    0
0      bar     one  1.0
1      bar     two  4.0
2      bar   three  NaN
3      bar    four  NaN
4      foo     one  NaN
5      foo     two  NaN
6      foo   three  1.0
7      foo    four  3.0
8      bar     one  2.0
9      bar     two  3.0
10     bar   three  NaN
11     bar    four  NaN
12     foo     one  NaN
13     foo     two  NaN
14     foo   three  NaN
15     foo    four  4.0
16     bar     one  2.0
17     bar     two  NaN
18     bar   three  NaN
19     bar    four  NaN
20     foo     one  NaN
21     foo     two  NaN
22     foo   three  NaN
23     foo    four  7.0
24     bar     one  3.0
25     bar     two  5.0
26     bar   three  NaN
27     bar    four  NaN
28     foo     one  NaN
29     foo     two  NaN
30     foo   three  NaN
31     foo    four  2.0
32     bar     one  4.0
33     bar     two  2.0
34     bar   three  NaN
35     bar    four  NaN
36     foo     one  NaN
37     foo     two  NaN
38     foo   three  3.0
39     foo    four  5.0
'''
# 实践表明，该方式不可行

# 3）方式3：使用转置和melt()

df4 = df.T.reset_index()
print(df4.to_string())
'''
  level_0 level_1    0    1    2    3    4
0     bar     one  1.0  2.0  2.0  3.0  4.0
1     bar     two  4.0  3.0  NaN  5.0  2.0
2     foo   three  1.0  NaN  NaN  NaN  3.0
3     foo    four  3.0  4.0  7.0  2.0  5.0
'''
df5 = df4.rename(columns={'level_0': 'a', 'level_1': 'c'}).melt(id_vars=['a', 'c'], value_name='v').drop(columns='variable')
print(df5.to_string())
'''
      a      c    v
0   bar    one  1.0
1   bar    two  4.0
2   foo  three  1.0
3   foo   four  3.0
4   bar    one  2.0
5   bar    two  3.0
6   foo  three  NaN
7   foo   four  4.0
8   bar    one  2.0
9   bar    two  NaN
10  foo  three  NaN
11  foo   four  7.0
12  bar    one  3.0
13  bar    two  5.0
14  foo  three  NaN
15  foo   four  2.0
16  bar    one  4.0
17  bar    two  2.0
18  foo  three  3.0
19  foo   four  5.0
'''


# 综合案例

# 读取数据
df_excel = pd.read_excel(r"C:\Users\cc\Desktop\mul_col_index.xlsx", header=[0, 1])
print(df_excel.to_string())
'''
  col1                col2                 
  name     city count name       city count
0    A  beijing    20    A   shanghai    25
1    B     xian    10    B  guangzhou    15
'''
# 指定某一层级列索引转换到行（列转行）
df_stack = df_excel.stack(level=0).reset_index(level=-1)
print(df_stack)
'''
  level_1 name       city  count
0    col1    A    beijing     20
0    col2    A   shanghai     25
1    col1    B       xian     10
1    col2    B  guangzhou     15
'''
# 全部层级列索引转换到行（列转行）
df_res = df_excel.T.reset_index()
print(df_res.to_string())
'''
  level_0 level_1         0          1
0    col1    name         A          B
1    col1    city   beijing       xian
2    col1   count        20         10
3    col2    name         A          B
4    col2    city  shanghai  guangzhou
5    col2   count        25         15
'''
df_res = df_res.melt(id_vars=['level_0', 'level_1'], value_name='v')
print(df_res.to_string())
'''
   level_0 level_1 variable          v
0     col1    name        0          A
1     col1    city        0    beijing
2     col1   count        0         20
3     col2    name        0          A
4     col2    city        0   shanghai
5     col2   count        0         25
6     col1    name        1          B
7     col1    city        1       xian
8     col1   count        1         10
9     col2    name        1          B
10    col2    city        1  guangzhou
11    col2   count        1         15
'''


# C、Python中的缺失值、缺失值检查
# Pandas能自动识别的Python缺失值字符串有：None、NA、nan、NaN、null、NULL、N/A、<NA>、''
# 不能自动识别的缺失值字符串有：na、Na、none、Null

print(np.NaN)             # NaN
print(type(np.NaN))       # <class 'float'>
print(pd.isnull(np.NaN))  # True
print(pd.isna(np.NaN))    # True

print(np.nan)             # NaN
print(type(np.nan))       # <class 'float'>
print(pd.isnull(np.nan))  # True
print(pd.isna(np.nan))    # True

print(pd.NA)              # <NA>
print(type(pd.NA))        # <class 'pandas._libs.missing.NAType'>
print(pd.isnull(pd.NA))   # True
print(pd.isna(pd.NA))     # True

# 时间格式的缺失值
print(pd.NaT)             # NaT
print(type(pd.NaT))       # <class 'pandas._libs.tslibs.nattype.NaTType'>
print(pd.isnull(pd.NaT))  # True
print(pd.isna(pd.NaT))    # True

print(None)               # None
print(type(None))         # <class 'NoneType'>
print(pd.isnull(None))    # True
print(pd.isna(None))      # True

# 空字符串不是缺失值
print('')                 #
print(type(''))           # <class 'str'>
print(pd.isnull(''))      # False
print(pd.isna(''))        # False

# Python如何检查缺失值（单个值、数组列表、Series）

df = pd.DataFrame({'key': [1, 2, 3, np.NaN], 'val': [0, False, None, np.NaN]})
print(df.to_string())
'''
   key    val
0  1.0      0
1  2.0  False
2  3.0   None
3  NaN    NaN
'''

# val列中是否包含缺失值（如NaN）
print(np.NaN in df.iloc[:, 1].tolist())   # True
print(np.NaN in df['val'].tolist())       # True
print(np.NaN in df['val'].values)         # False
print(True in df['val'].isnull().values)  # True

# 方法1）np.isnan()
# 可检查单个值、数组列表对象的缺失值（仅限Number类型）
# 单个值返回True或False；类似数组的对象返回一个与输入形状相同的布尔数组
print(np.isnan(np.NaN))    # True
# 由于None的类型为NoneType（非Number类型），所以None或包含None的数组列表执行将报错
# print(np.isnan(None))    # 报错
# print(np.isnan(df['val'].values))  # 报错
print(np.isnan(df['key'].values))    # [False False False True]

# 方法2）pd.isna()
# 可检查单个值、数组列表、Pandas数据结构（如Series、DataFrame）对象的缺失值（NaN、None或NaT）
# 单个值返回True或False；类似数组的对象返回一个与输入形状相同的布尔数组
print(pd.isna([None]))     # [True]
print(pd.isna(np.NaN))     # True
print(pd.isna(pd.NaT))     # True
print(pd.isna(df['val'].values))     # [False False True True]






import numpy as np
import pandas as pd

'''
12、stack()、unstack()
'''
# 1）stack(Level=-1, dropna=True)堆叠：将DataFrame多层列压缩至一层，压缩后的DataFrame具有多层索引
'''
Level=-1：默认第二层列；dropna=True：默认删除所有值都缺失的行
'''
df = pd.DataFrame({'C1': ['R1', 'R2'], 'C2': ['V1', 'V2'], 'C3': ['V3', 'V4']})
# print(df.to_string())
'''
   C1  C2  C3
0  R1  V1  V3
1  R2  V2  V4
'''
df.set_index(['C1'], inplace=True)

df1 = df.stack().reset_index()
# print(df1.to_string())   # 列转行效果
'''
   C1 level_1   0
0  R1      C2  V1
1  R1      C3  V3
2  R2      C2  V2
3  R2      C3  V4
'''

# 2）unstack(Level=-1, fill_value=None)拆叠：与stack()相反
'''
Level=-1：默认第二层索引；fill_value=None：匹配不到值使用None填充
'''
df = pd.DataFrame({'C1': ['R1', 'R2'], 'C2': ['T', 'W'], 'C3': ['V1', 'V2']})
# print(df.to_string())
'''
   C1 C2  C3
0  R1  T  V1
1  R2  W  V2
'''
df.set_index(['C1'], inplace=True)
# print(df.to_string())
'''
   C2  C3
C1       
R1  T  V1
R2  W  V2
'''
df2 = df.unstack().reset_index()
# print(df2.to_string())   # 列转行效果
'''
  level_0  C1   0
0      C2  R1   T
1      C2  R2   W
2      C3  R1  V1
3      C3  R2  V2
'''

# 案例1：处理多行字段名的Excel表
# 读取Excel文件，并指定标题行数为2，header=[0, 1]表示前两行被用作列名
# df = pd.read_excel('excel', sheet_name='sheet', header=[0, 1])

# 案例2：stack()、unstack()
df = pd.DataFrame({('Level1', 'A'): [1, 2, 3], ('Level1', 'B'): [4, 5, 6]})
# print(df.to_string())
'''
     Level1   
       A  B
0      1  4
1      2  5
2      3  6
'''
# 输出原始的多级索引列名
# print(df.columns)
'''
MultiIndex([('Level1', 'A'),('Level1', 'B')],)
'''
# print(df.stack())    # Level=-1，取第二层列字段A和B
'''
     Level1
0 A       1
  B       4
1 A       2
  B       5
2 A       3
  B       6
'''
# print(df.stack().reset_index(level=-1))   # Level=-1，取第二层索引A和B
'''
  level_1  Level1
0       A       1
0       B       4
1       A       2
1       B       5
2       A       3
2       B       6
'''
# print(df.stack().unstack())
'''
  Level1   
       A  B
0      1  4
1      2  5
2      3  6
'''

# print(df.stack(0))    # Level=0，取第一层列字段Level1
'''
          A  B
0 Level1  1  4
1 Level1  2  5
2 Level1  3  6
'''
# print(df.stack(0).reset_index(level=-1))   # Level=-1，取第二层索引Level1
'''
  level_1  A  B
0  Level1  1  4
1  Level1  2  5
2  Level1  3  6
'''
# print(df.stack(0).unstack())
'''
       A      B
  Level1 Level1
0      1      4
1      2      5
2      3      6
'''

# 案例3：DataFrame多行字段名变成单行字段名（将多级索引的列名转换为单行字段名）
df.columns = df.columns.map('.'.join)
# 重置索引，并将列名作为新的索引值
df = df.reset_index(drop=True)
# print(df.to_string())
'''
   Level1.A  Level1.B
0         1         4
1         2         5
2         3         6
'''
# 输出转换后的单行字段名
# print(df.columns)
'''
Index(['Level1.A', 'Level1.B'], dtype='object')
'''

'''
13、窗口函数
'''
# 1）rolling()滑动窗口函数
'''
rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
  window：窗口大小，取值类型：1）int：固定窗口大小，包含相同数量的观测值；2）offset：包含不确定数量的观测值
  min_periods：窗口最少包含的观测值，取值类型：1）int：默认None；2）offset：默认1
  center：是否将窗口标签设置为居中，默认False
  win_type：窗口类型
  on：指定df列进行窗口操作
  axis：进行窗口操作的轴，0对列，1对行
  closed：表示定义窗口区间的开闭
对时间序列数据进行窗口操作，并计算窗口内的统计量
  count()             统计非空数量
  sum()	              求和
  mean()	          平均值
  median()	          中位数
  min()、max()	      最小值、最大值
  std()	              标准差
  var()     	      方差
  apply()	          apply函数使用
  agg()               agg函数使用
'''
df = pd.DataFrame({"sale": np.arange(10)}, index=pd.date_range('2022-01-01', periods=10))
print(df.to_string())
'''
            sale
2022-01-01     0
2022-01-02     1
2022-01-03     2
2022-01-04     3
2022-01-05     4
2022-01-06     5
2022-01-07     6
2022-01-08     7
2022-01-09     8
2022-01-10     9
'''
# 案例：计算近3天的总量和平均值

df['sum'] = df['sale'].rolling(3).sum()
df['avg'] = df['sale'].rolling(3).mean()
# min_periods：必须≤window大小，对第n个元素往前数至少满足min_periods个值进行求平均值
df['sum_min_per'] = df['sale'].rolling(3, min_periods=1).sum()
df['avg_min_per'] = df['sale'].rolling(3, min_periods=2).mean()
# center：以当前元素为中心，在上下两个方向进行窗口的统计计算
df['sum_center'] = df['sale'].rolling(3, center=True).sum()
df['avg_center'] = df['sale'].rolling(3, center=True).mean()
# closed：right去除窗口中第一个数据；left去除窗口中最后一个数据；neither去除窗口中第一个和最后一个数据；both不去除任何数据；

# 应用多个函数
df[['sum', 'mean', 'std']] = df.rolling(window=3, min_periods=1)['sale'].agg([np.sum, np.mean, np.std])
# 不同列应用不同函数
df[['sum_sum', 'mean_std']] = df.rolling(window=3, min_periods=1).agg({'sum': np.sum, 'mean': lambda x: np.std(x, ddof=1)})
print(df.to_string())
'''
            sale   sum  mean       std  sum_sum  mean_std
2022-01-01     0   0.0   0.0       NaN      0.0       NaN
2022-01-02     1   1.0   0.5  0.707107      1.0  0.353553
2022-01-03     2   3.0   1.0  1.000000      4.0  0.500000
2022-01-04     3   6.0   2.0  1.000000     10.0  0.763763
2022-01-05     4   9.0   3.0  1.000000     18.0  1.000000
2022-01-06     5  12.0   4.0  1.000000     27.0  1.000000
2022-01-07     6  15.0   5.0  1.000000     36.0  1.000000
2022-01-08     7  18.0   6.0  1.000000     45.0  1.000000
2022-01-09     8  21.0   7.0  1.000000     54.0  1.000000
2022-01-10     9  24.0   8.0  1.000000     63.0  1.000000
'''

# 2）expanding()扩展窗口函数
'''
由序列的第一个元素开始，逐个向后计算元素的统计量
'''
df['sum_expand'] = df['sale'].expanding(3).sum()
df['avg_expand'] = df['sale'].expanding(3).mean()
print(df.to_string())
'''
            sale   sum  avg  sum_min_per  avg_min_per  sum_center  avg_center  sum_expand  avg_expand
2022-01-01     0   NaN  NaN          0.0          NaN         NaN         NaN         NaN         NaN
2022-01-02     1   NaN  NaN          1.0          0.5         3.0         1.0         NaN         NaN
2022-01-03     2   3.0  1.0          3.0          1.0         6.0         2.0         3.0         1.0
2022-01-04     3   6.0  2.0          6.0          2.0         9.0         3.0         6.0         1.5
2022-01-05     4   9.0  3.0          9.0          3.0        12.0         4.0        10.0         2.0
2022-01-06     5  12.0  4.0         12.0          4.0        15.0         5.0        15.0         2.5
2022-01-07     6  15.0  5.0         15.0          5.0        18.0         6.0        21.0         3.0
2022-01-08     7  18.0  6.0         18.0          6.0        21.0         7.0        28.0         3.5
2022-01-09     8  21.0  7.0         21.0          7.0        24.0         8.0        36.0         4.0
2022-01-10     9  24.0  8.0         24.0          8.0         NaN         NaN        45.0         4.5
'''

# 3）ewm()指数加权滑动窗口
'''
ewn()函数先会对序列元素做指数加权运算，然后计算加权后的统计量
'''

'''
14、pivot_table()数据透视表
'''
# 数据透视表（Pivot Table）是数据分析中常见的工具之一，根据一个或多个键值对数据进行聚合，根据列或行的分组键将数据划分到各个区域
# 在Pandas中，除了使用groupby对数据分组聚合实现透视功能外，还可以使用pivot_table函数实现

# pd.pivot_table() 或 df.pivot_table()
'''
pivot_table(data, values, index, columns, aggfunc='mean', fill_value, margins=False, dropna=True, margins_name='All')
- data：DataFrame对象
- values：需要计算的数据列
- index：行层次的分组键，从而形成多级索引，str或list类型
- columns：列层次的分组键，从而形成多级列名，str或list类型
- aggfunc：对数据执行的聚合操作函数，默认mean计算平均（支持应用多个函数）
- fill_value：设定缺失替换值
- margins：是否添加行列的汇总
- dropna：如果列的所有值都为NaN，将不作为计算列，默认True
- margins_name：汇总行列的名称，默认为All
'''
df = pd.DataFrame({
    'area': ['A', 'B', 'A', 'A', 'C', 'B'],
    'city': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
    'sale': [24, 19, 35, 11, 22, 18]
})
print(df.to_string())
'''
  area city  sale
0    A   c1    24
1    B   c2    19
2    A   c3    35
3    A   c4    11
4    C   c5    22
5    B   c6    18
'''

# 计算各地区总销量
df1 = pd.pivot_table(df, index='area', values=['sale'], aggfunc=np.sum).reset_index()
print(df1.to_string())
'''
  area  sale
0    A    70
1    B    37
2    C    22
'''

# 计算每个地区每个城市的平均销量
df2 = pd.pivot_table(df, index=['area', 'city'], values=['sale'], aggfunc=np.mean).reset_index()
print(df2.to_string())
'''
  area city  sale
0    A   c1    24
1    A   c3    35
2    A   c4    11
3    B   c2    19
4    B   c6    18
5    C   c5    22
'''

# 计算各个地区各个城市各个种类的销量情况
df = pd.DataFrame({
    'area': ['A', 'B', 'A', 'A', 'C', 'B'],
    'city': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
    'category': ['one', 'tow', 'tow', 'one', 'tow', 'tow'],
    'sale': [24, 19, 35, 11, 22, 18]
})
print(df.to_string())
'''
  area city category  sale
0    A   c1      one    24
1    B   c2      tow    19
2    A   c3      tow    35
3    A   c4      one    11
4    C   c5      tow    22
5    B   c6      tow    18
'''
df3 = pd.pivot_table(df, index=['area', 'city'], values='sale', columns=['category'])
print(df3.to_string())
'''
category    one   tow
area city            
A    c1    24.0   NaN
     c3     NaN  35.0
     c4    11.0   NaN
B    c2     NaN  19.0
     c6     NaN  18.0
C    c5     NaN  22.0
'''
df3 = df3.reset_index()
print(df3.to_string())
'''
category area city   one   tow
0           A   c1  24.0   NaN
1           A   c3   NaN  35.0
2           A   c4  11.0   NaN
3           B   c2   NaN  19.0
4           B   c6   NaN  18.0
5           C   c5   NaN  22.0
'''

# pivot()：若存在多级索引，则会报错，可以使用pd.pivot_table()
df_pivot = df.pivot(index='city', columns='area', values='sale').reset_index()
print(df_pivot.to_string())
'''
area city     A     B     C
0      c1  24.0   NaN   NaN
1      c2   NaN  19.0   NaN
2      c3  35.0   NaN   NaN
3      c4  11.0   NaN   NaN
4      c5   NaN   NaN  22.0
5      c6   NaN  18.0   NaN
'''

# 15、crosstab()交叉制表
# crosstab(index,columns,values,aggfunc)
'''
- index：索引
- columns：列名
- values：根据聚合因子(aggfunc)计算后的值，若不指定，则默认计算因子的频率表
- aggfunc：函数，聚合因子
'''
df_crosstab = pd.crosstab(index=df.city, columns=df.area, values=df.sale, aggfunc=lambda x: x)
print(df_crosstab.to_string())
'''
area     A     B     C
city                  
c1    24.0   NaN   NaN
c2     NaN  19.0   NaN
c3    35.0   NaN   NaN
c4    11.0   NaN   NaN
c5     NaN   NaN  22.0
c6     NaN  18.0   NaN
'''

# 16、pipe()管道函数
# 可用于多列组合运算与链式运算
df = pd.DataFrame({
    'store': ['S1', 'S2', 'S2', 'S1'],
    'product': ['P2', 'P2', 'P1', 'P1'],
    'revenue': (np.random.random(4) * 10).round(2),
    'quantity': np.random.randint(1, 10, size=4)
})
print(df.to_string())
'''
  store product  revenue  quantity
0    S1      P2     6.49         5
1    S2      P2     0.50         9
2    S2      P1     5.32         2
3    S1      P1     2.30         2
'''
# 计算每个商店每个产品的平均价格（收入/数量）
df_pipe = df.groupby(['store', 'product']).pipe(lambda g: g.revenue.sum() / g.quantity.sum()).round(2)
print(df_pipe.to_string())
'''
store  product
S1     P1         1.15
       P2         1.30
S2     P1         2.66
       P2         0.06
'''
print(df_pipe.unstack(level=-1).reset_index())
'''
product store    P1    P2
0          S1  1.15  1.30
1          S2  2.66  0.06
'''
# 将分组对象传递给某个任意函数（可自定义）
def mean(g):
    return g.mean()

print(df.groupby(['store', 'product']).pipe(mean))
print(df.groupby(['store', 'product']).pipe(lambda g: g.mean()))
'''
               revenue  quantity
store product                   
S1    P1          2.30       2.0
      P2          6.49       5.0
S2    P1          5.32       2.0
      P2          0.50       9.0
'''

# 17）Pandas的增强功能
df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 5]})
print(df.to_string())
'''
   a  b
0  1  2
1  2  3
2  3  5
'''
# df.eval()表达式运算
df.eval('c = a + b', inplace=True)
print(df.to_string())
'''
   a  b  c
0  1  2  3
1  2  3  5
2  3  5  8
'''
# df.query()表达式筛选
print(df.query('b > 3'))
'''
   a  b  c
2  3  5  8
'''
print(df.query('a > 1 and b < 5'))
'''
   a  b  c
1  2  3  5
'''



# pandas相关函数示例

import numpy as np
import pandas as pd

'''
10、行转列、列转行
'''
df = pd.DataFrame({'dt': [20, 21], 'A': [80, 90], 'B': [82, 88], 'C': [91, 79]})
# print(df.to_string())
df1 = pd.DataFrame({'name': ['a', 'b'], 'hobby': ['lq, cg', 'jt, ymq, ps']})
print(df1.to_string())
'''
   dt   A   B   C
0  20  80  82  91
1  21  90  88  79
'''
'''
  name        hobby
0    a       lq, cg
1    b  jt, ymq, ps
'''
# 1）列转行
# A、列转行1
# 方法1：pd.melt()
'''
id_vars: 不需要做列转行处理的字段，如果不设置该字段则默认会对所有列进行处理
var_name: 列转行处理后，生成字段列，对列转行之前的字段名称进行重命名
value_name: 列转行处理后，生成数值列，对列转行之前的数值进行命名
value_vars: 需要做列转行的字段，不指定则不处理
'''
df_melt = pd.melt(df, id_vars=['dt'], var_name='country', value_name='value')
# print(df_melt.to_string())
'''
   dt country  value
0  20       A     80
1  21       A     90
2  20       B     82
3  21       B     88
4  20       C     91
5  21       C     79
'''
# 方法2：
# set_index(['col']) 将某列值设置为行索引
# stack(list,axis=0) 将一个列表转换为一个numpy数组，当axis=0时，和np.array()没有区别；当axis=1时，对每一行在列方向上进行运算，即将矩阵的维度从（m,n）变成（n,m）
# reset_index() 将原来的索引index作为新的一列（添加了一列）
tmp = df.set_index(['dt']).stack().reset_index()
tmp.columns = ['dt', 'country', 'value']
# print(tmp.to_string())
'''
   dt country  value
0  20       A     80
1  20       B     82
2  20       C     91
3  21       A     90
4  21       B     88
5  21       C     79
'''
# B、列转行2
# 方法1：
df2 = df1.set_index(['name'])
# split():
df2 = df2['hobby'].str.split(',', expand=True)
df2 = df2.rename_axis(columns=None).reset_index()
tmp = df2.set_index(['name']).stack().reset_index()
tmp.columns = ['name', 'index', 'hobby']
tmp = tmp[['name', 'hobby']]
# print(tmp.to_string())
'''
  name hobby
0    a    lq
1    a    cg
2    b    jt
3    b   ymq
4    b    ps
'''
# 方法2：
df1['hobby'] = df1['hobby'].str.split(',')
# df.explode(arr_col) 列转行
df1 = df1.explode('hobby')
# print(df1.to_string())
'''
  name hobby
0    a    lq
0    a    cg
1    b    jt
1    b   ymq
1    b    ps
'''

# explode()：若arr_col中存在空列表[]、空值None或NaN，转换结果也为None或NaN，数据不会丢失
df = pd.DataFrame({'id': ['01', '02', '03', '04', '05'], 'arr': [[1, 3], None, [], 10, np.nan]})
# print(df.to_string())
'''
   id     arr
0  01  [1, 3]
1  02    None
2  03      []
3  04      10
4  05     NaN
'''
# print(df.explode('arr'))
'''
   id   arr
0  01     1
0  01     3
1  02  None
2  03   NaN
3  04    10
4  05   NaN
'''

# 案例：列转行，多个字段不动（不变）
df = pd.DataFrame({'Y': [2020, 2021, 2022], 'X': ['01', '02', '03'], 'A': [10, 20, 15], 'B': [20, 25, 30]})
# print(df.to_string())
'''
      Y   X   A   B
0  2020  01  10  20
1  2021  02  20  25
2  2022  03  15  30
'''
# Y、X列保持原来
df_melt = pd.melt(df, id_vars=['Y', 'X'], var_name='C', value_name='V')
# print(df_melt.to_string())
'''
      Y   X  C   V
0  2020  01  A  10
1  2021  02  A  20
2  2022  03  A  15
3  2020  01  B  20
4  2021  02  B  25
5  2022  03  B  30
'''

# 2）行转列
# A、pivot(index=None,columns=None,values=None): index用作行的列名 columns用作列的列名 values用作值的列名
# rename_axis(mapper=None,index=None,columns=None,axis=0,inplace=False) 设置索引或列的axis名称:
# mapper类似list/optional index/columns行/列 axis:0或index,1或columns
df_pivot = df_melt.pivot(index='dt', columns='country', values='value').rename_axis(columns=None).reset_index()
# print(df_pivot.to_string())
'''
   dt   A   B   C
0  20  80  82  91
1  21  90  88  79
'''

# B、groupby('col').agg/aggregate(list/set) 分组收集
df = pd.DataFrame({'name': ['A', 'B', 'A', 'B', 'C'], 'value': [1, 2, 3, 4, 5]})
# print(df.to_string())
'''
  name  value
0    A      1
1    B      2
2    A      3
3    B      4
4    C      5
'''
df1 = df.groupby(['name']).agg(list).reset_index()
# print(df1.to_string())
'''
  name   value
0    A  [1, 3]
1    B  [2, 4]
2    C     [5]
'''

'''
11、groupby分组、agg()聚合
'''
'''
groupby(by=['col1',...],as_index=True,group_keys=True,) 按指定列对df分组（支持多级分组）
by=[cols]：分组列、分组键
as_index=True（默认）：分组的键默认显示在索引中，False可取消
group_keys=True（默认）：控制索引中是否包含分组列，默认包含，False可选择不包含（相当于as_index=False）
sort=True（默认）：按分组列的结果按升序排序，False将结果按降序排序
'''
# 注意：分组键中存在NaN：会自动丢弃这些值，即永远不会有NaN组
df = pd.DataFrame({'Y': [2020, 2021, 2020], 'C': ['01', '01', '02'], 'V': [20, 25, 10]})
# print(df.to_string())
'''
      Y   C   V
0  2020  01  20
1  2021  01  25
2  2020  02  10
'''
# 1）查看分组结果
grouped = df.groupby(by='Y')     # 分组对象
# 循环访问组
for name, group in grouped:
    print(name)      # 组名（按多个键分组后，组名是元组类型）
    print(group)     # 每组数据(df)

# 查看分组结果（以字典形式显示）
print(grouped.groups)
# 查看分组结果（以df形式显示）
print(grouped.head())
# 选择组：get_group(组名)
print(grouped.get_group(2020))

# 2）除分组列其它所有列都会参与聚合函数运算
df_sum = df.groupby(by='Y').sum().reset_index()       # 方式1
# print(df_sum.to_string())
# 分组键会默认显示在索引中，as_index=False可取消
df_sum = df.groupby(by='Y', as_index=False).sum()     # 方式2
# print(df_sum.to_string())
'''
      Y     C   V
0  2020  0102  30
1  2021    01  25
'''

# 3）指定列参与聚合函数运算（其它列不会在结果df中显示）
df_sum1 = df.groupby(by='Y')['V'].sum().reset_index()       # 方式1
# print(df_sum1.to_string())
df_sum2 = df.groupby(by='Y').sum('V').reset_index()         # 方式2
# print(df_sum2.to_string())
df_sum3 = df.groupby(by='Y')['V'].agg('sum').reset_index()  # 方式3
# print(df_sum3.to_string())
'''
      Y   V
0  2020  30
1  2021  25
'''
df_sum4 = df.groupby(by='Y')['C'].agg(lambda x: ','.join(x)).reset_index()
# print(df_sum4.to_string())
'''
      Y      C
0  2020  01,02
1  2021     01
'''
df_sum5 = df.groupby(by='Y')['C'].agg(lambda x: list(x)).reset_index()
# print(df_sum5.to_string())
'''
      Y         C
0  2020  [01, 02]
1  2021      [01]
'''

# A、若每列需要作不同计算，可单独计算再合并（方式1）
df_merge = pd.merge(df_sum4, df_sum1, on='Y')
# print(df_merge.to_string())
'''
      Y      C   V
0  2020  01,02  30
1  2021     01  25
'''

# agg()聚合

# 4）一次应用多个函数
df_sum6 = df.groupby(by='Y').agg(['sum', lambda y: list(set(y))]).reset_index()
# print(df_sum6.to_string())
'''
      Y     C              V           
          sum <lambda_0> sum <lambda_0>
0  2020  0102   [02, 01]  30   [10, 20]
1  2021    01       [01]  25       [25]
'''

# 生成的聚合结果列以函数本身命名
# 重命名：使用rename
df_sum6 = df.groupby(by='Y').agg(['sum', lambda y: list(set(y))]).reset_index().rename(columns={'': 'Y', 'sum': 'V_sum', '<lambda_0>': 'values'})
print(df_sum6.to_string())
'''
      Y     V          
      Y V_sum    values
0  2020    30  [10, 20]
1  2021    25      [25]
'''
print(df_sum6.columns.tolist())     # [('Y', 'Y'), ('V', 'V_sum'), ('V', 'values')]
# 使用第二层列索引作为列名
df_sum6.columns = [t[1] for t in df_sum6.columns.tolist()]
print(df_sum6.to_string())
'''
      Y  V_sum    values
0  2020     30  [10, 20]
1  2021     25      [25]
'''

# B、若每列需要作不同计算，可对不同列应用不同函数（方式2）
df_sum7 = df.groupby(by='Y').agg({'C': list, 'V': 'sum'}).reset_index()
# print(df_sum7.to_string())
'''
      Y         C   V
0  2020  [01, 02]  30
1  2021      [01]  25
'''
# 一次应用多个函数
df_sum7 = df.groupby(by='Y').agg({'C': [list, 'sum'], 'V': ['sum', 'mean']}).reset_index()
print(df_sum7.to_string())
'''
      Y         C         V      
             list   sum sum  mean
0  2020  [01, 02]  0102  30  15.0
1  2021      [01]    01  25  25.0
'''

# C、若每列需要作不同计算，可对不同列应用不同函数并重命名（方式3）
df_sum8 = df.groupby(by='Y').agg(C_sum=('C', lambda y: list(set(y))), V_sum=('V', 'sum')).reset_index()
# print(df_sum8.to_string())
'''
      Y     C_sum  V_sum
0  2020  [02, 01]     30
1  2021      [01]     25
'''

# 5）分组后按聚合结果排序
df = pd.DataFrame({'Y': [2021, 2020, 2021], 'C': ['01', '01', '02'], 'V': [20, 25, 10]})
df1 = df.groupby(by='Y', sort=True)['V'].sum().reset_index()
# print(df1.to_string())    # sort=True 默认按结果升序排序
'''
      Y   V
0  2020  25
1  2021  30
'''
df2 = df.groupby(by='Y', sort=False)['V'].sum().reset_index()
# print(df2.to_string())    # sort=False 按结果降序排序
'''
      Y   V
0  2021  30
1  2020  25
'''

# 6）组内排序、组内计算排行
# 6.1）组内排序：apply(lambda+sort_values())
# 分组后组内按指定字段排序（降序） group_keys=False：控制索引中不包含分组列
df_t1 = df.groupby(by='Y', group_keys=False).apply(lambda x: x.sort_values('V', ascending=False))
# print(df_t1.to_string())
'''
      Y   C   V
0  2021  01  20
1  2020  01  25
2  2021  02  10
'''
# 6.2）组内计算排行：apply(lambda+rank())
df['V_rk'] = df.groupby(by='Y', group_keys=False).apply(lambda x: x['V'].rank(method='dense', ascending=False))
# print(df.to_string())
'''
      Y   C   V  V_rk
0  2021  01  20   1.0
1  2020  01  25   1.0
2  2021  02  10   2.0
'''

# 7）组内过滤、组内筛选
df = pd.DataFrame({"dt": pd.date_range(start="2018-01-01", periods=5, freq="W"), "cla": ['B', 'B', 'M', 'M', 'M'], "v": [7, 6, 8, 5, 6]})
# print(df.to_string())
'''
          dt cla  v
0 2018-01-07   B  7
1 2018-01-14   B  6
2 2018-01-21   M  8
3 2018-01-28   M  5
4 2018-02-04   M  6
'''
# 7.1）nth(n) 筛取组内第n行，n从0开始
# 取组内第n个 nth(n, dropna): n: 可为数组，取多个行  dropna: any、all，可去除空值行
# a、先进行组内排序
df = df.groupby('cla', group_keys=False).apply(lambda x: x.sort_values('v'))
# b、再取组内排序后的第一个（每组第一）
# print(df.groupby('cla').nth(0))
'''
          dt cla  v
1 2018-01-14   B  6
3 2018-01-28   M  5
'''
# 7.2）head()、tail() 筛选分组后的df前n或后n行
# 函数补充：first()：取组内第一个 last()：取组内最后一个
# print(df.groupby('cla').head(1))      # 每组第一
'''
          dt cla  v
1 2018-01-14   B  6
3 2018-01-28   M  5
'''
# print(df.groupby('cla').tail(1))      # 每组倒一
'''
          dt cla  v
0 2018-01-07   B  7
2 2018-01-21   M  8
'''
# 7.3）filter()过滤
df = pd.DataFrame({"dt": pd.date_range(start="2018-01-01", periods=5, freq="W"), "cla": ['B', 'B', 'M', 'M', 'M'], "v": [7, 6, 8, 5, 6]})
df_f = df.groupby('cla').filter(lambda x: len(x) > 2)
# print(df_f.to_string())
'''
          dt cla  v
2 2018-01-21   M  8
3 2018-01-28   M  5
4 2018-02-04   M  6
'''

# 8）组内计算同比环比
# 8.1）生成新列追加到df
# df['同比'] = df.groupby('cla')['v'].pct_change()
# print(df.to_string())
'''
          dt cla  v        同比
0 2018-01-07   B  7       NaN
1 2018-01-14   B  6 -0.142857
2 2018-01-21   M  8       NaN
3 2018-01-28   M  5 -0.375000
4 2018-02-04   M  6  0.200000
'''

# 8.2）指定列计算同比纵向拼接
df_index = df.set_index(['dt', 'cla'])
df_index = df_index[df_index.columns].pct_change().reset_index()
# print(df_index.to_string())
''' 
          dt cla         v
0 2018-01-07   B       NaN
1 2018-01-14   B -0.142857
2 2018-01-21   M  0.333333
3 2018-01-28   M -0.375000
4 2018-02-04   M  0.200000
'''
# 纵向拼接
df_concat = pd.concat([df, df_index])
# print(df_concat.to_string())

# groupby()补充：

# 1）根据level分组（多级索引分组）
df = pd.DataFrame({'Y': [2020, 2021, 2020], 'C': ['01', '01', '02'], 'V': [20, 25, 10]}).set_index(['Y', 'C'])
print(df.to_string())
'''
          V
Y    C     
2020 01  20
2021 01  25
2020 02  10
'''
df1 = df.groupby(level='Y', as_index=False).sum()
df1 = df.groupby(level=0).sum()
print(df1.to_string())
'''
       V
Y       
2020  30
2021  25
'''
df2 = df.groupby(level=[0, -1]).sum()
print(df2.to_string())
'''
          V
Y    C     
2020 01  20
     02  10
2021 01  25
'''
# 2）使用索引级别和列进行分组
arrays = [['bar', 'bar', 'foo', 'foo'], ['one', 'two', 'one', 'two']]
dfx = pd.DataFrame({'A': [1, 1, 2, 2], 'B': np.arange(4)}, index=pd.MultiIndex.from_arrays(arrays, names=['first', 'second']))
print(dfx.to_string())
'''
              A  B
first second      
bar   one     1  0
      two     1  1
foo   one     2  2
      two     2  3
'''
dfx1 = dfx.groupby([pd.Grouper(level=-1), 'A']).sum()
dfx1 = dfx.groupby(['second', 'A']).sum()
print(dfx1)
'''
          B
second A   
one    1  0
       2  2
two    1  1
       2  3
'''
# 3）选择组：get_group(组名)
print(dfx.groupby(['second', 'A']).get_group(('one', 2)).reset_index())
'''
  first second  A  B
0   foo    one  2  2
'''
# 4）lambda分组：lambda参数默认为行索引
df = pd.DataFrame({"dt": pd.date_range(start="2022-12-01", periods=5, freq="W"), "cla": ['B', 'B', 'M', 'M', 'M'], "v": [7, 6, 8, 5, 6]}).set_index('dt')
print(df.to_string())
'''
           cla  v
dt               
2022-12-04   B  7
2022-12-11   B  6
2022-12-18   M  8
2022-12-25   M  5
2023-01-01   M  6
'''
# 指定一个列参与聚合函数运算（默认返回Series）
df_sum = df.groupby(lambda x: x.year)['v'].sum().reset_index()
df_sum = df.groupby(lambda x: x.year).v.sum().reset_index()
print(df_sum.to_string())
'''
     dt   v
0  2022  26
1  2023   6
'''
# 5）groupby()后使用窗口函数
# 此处df见4）中
df_mean = df.groupby('cla').rolling(2).v.mean().reset_index()
print(df_mean.to_string())
'''
  cla         dt    v
0   B 2022-12-04  NaN
1   B 2022-12-11  6.5
2   M 2022-12-18  NaN
3   M 2022-12-25  6.5
4   M 2023-01-01  5.5
'''
df_sum = df.groupby('cla').expanding().v.sum().reset_index()
print(df_sum.to_string())
'''
  cla         dt     v
0   B 2022-12-04   7.0
1   B 2022-12-11  13.0
2   M 2022-12-18   8.0
3   M 2022-12-25  13.0
4   M 2023-01-01  19.0
'''
# 6）groupby()后使用transform()
# 此处df_sum见5）中
df_trans = df_sum.groupby('cla').transform(lambda x: x.mean())
print(df_trans)
'''
                   dt          v
0 2022-12-07 12:00:00       10.0
1 2022-12-07 12:00:00       10.0
2 2022-12-25 00:00:00  13.333333
3 2022-12-25 00:00:00  13.333333
4 2022-12-25 00:00:00  13.333333
'''
# 指定列生成新列：groupby(cols)['col'] 或 groupby(cols).col
df_sum['v_mean'] = df_sum.groupby('cla').v.transform(lambda x: x.mean())
print(df_sum.to_string())
'''
  cla         dt     v     v_mean
0   B 2022-12-04   7.0  10.000000
1   B 2022-12-11  13.0  10.000000
2   M 2022-12-18   8.0  13.333333
3   M 2022-12-25  13.0  13.333333
4   M 2023-01-01  19.0  13.333333
'''
# 7）按时间频率分组 pd.Grouper(key,level,freq)
# 此处df见4）中
df_r = df.reset_index()
print(df_r.to_string())
# freq（M\Q\Y显示为最后一天）: D按天 W按周 M按月 Q按季度 Y按年
df_g = df_r.groupby([pd.Grouper(freq='Y', key='dt'), 'cla']).sum().reset_index()
print(df_g.to_string())
'''
          dt cla   v
0 2022-12-31   B  13
1 2022-12-31   M  13
2 2023-01-31   M   6
'''
# M\Q\Y若要显示第一天，可使用apply()操作

# 8）groupby()后使用apply()
import datetime
df_g1 = df_r.groupby([df_r['dt'].apply(lambda x: datetime.datetime(x.year, x.month, 1)), 'cla']).v.sum().reset_index()
print(df_g1.to_string())
'''
          dt cla   v
0 2022-12-01   B  13
1 2022-12-01   M  13
2 2023-01-01   M   6
'''
# 计算每个种类cla的所有月份和平均值
df_apply = df_r.groupby('cla').apply(lambda x: pd.Series({'month_list': list(set(x['dt'].apply(lambda y: y.month))), 'v_mean': x['v'].mean().round(2)}))
print(df_apply.to_string())
'''
    month_list  v_mean
cla                   
B         [12]    6.50
M      [1, 12]    6.33
'''




import pandas as pd
import numpy as np

# 多级/分层索引（MultiIndex）、时间序列与日期

# A、多级/分层索引（MultiIndex）

# 多级行索引
# 1）创建
# a、从数组列表：MultiIndex.from_arrays()
df = pd.DataFrame(np.random.randn(4), columns=['v'])
print(df.to_string())
'''
          v
0 -1.066128
1  0.393179
2  1.595051
3 -1.492583
'''
arrays = [np.array(['bar', 'bar', 'foo', 'foo']), np.array(['one', 'two', 'one', 'two'])]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df.index = index
print(df.to_string())
'''
                     v
first second          
bar   one     1.270260
      two    -0.539714
foo   one    -0.284528
      two     0.682699
'''
# b、从元组数组：MultiIndex.from_tuples()
df = pd.DataFrame(np.random.randn(4), columns=['v'])
arrays = [['bar', 'bar', 'foo', 'foo'], ['one', 'two', 'one', 'tow']]
tuples = list(zip(*arrays))
# print(tuples)    # [('bar', 'one'), ('bar', 'two'), ('foo', 'one'), ('foo', 'tow')]
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df.index = index
# print(df.to_string())
'''
                     v
first second          
bar   one    -1.066128
      two     0.393179
foo   one     1.595051
      tow    -1.492583
'''
# c、从交叉迭代器集（两两匹配）：MultiIndex.from_product()
df = pd.DataFrame(np.random.randn(4), columns=['v'])
iterables = [['bar', 'foo'], ['one', 'two']]
index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
df.index = index
# print(df.to_string())
'''
                     v
first second          
bar   one    -0.201214
      two    -1.097793
foo   one     0.033961
      two    -0.023977
'''

# d、从DataFrame：MultiIndex.from_frame()
df = pd.DataFrame(np.random.randn(4), columns=['v'])
df_index = pd.DataFrame([['bar', 'one'], ['bar', 'two'], ['foo', 'one'], ['foo', 'two']], columns=['first', 'second'])
index = pd.MultiIndex.from_frame(df_index)
df.index = index
# print(df.to_string())
'''
                     v
first second          
bar   one    -0.669057
      two     0.749700
foo   one    -1.458682
      two    -0.795187
'''

# 2）获取指定层索引标签列表 get_level_values()
print(df.index.get_level_values(-1))    # Index(['one', 'two', 'one', 'two'], dtype='object', name='second')

# 3）数据对齐 reindex()
# 调整列（默认行）：可添加列、对列调换位置
df_reindex = df.reindex(['k', 'v'], axis=1, fill_value=0)
# print(df_reindex.to_string())
'''
              k         v
first second             
bar   one     0  0.778402
      two     0 -0.773582
foo   one     0  0.102783
      two     0 -0.849725
'''

# 4）多级/分层索引高级用法
# 取指定索引的所有列的值
# print(df_reindex.loc[('bar', 'two')])
'''
k    0.000000
v   -0.115918
Name: (bar, two), dtype: float64
'''
# 取指定索引指定列的值
# print(df_reindex.loc[('bar', 'two'), 'v'])
'''
1.1971089038835172
'''
# 局部切片
# print(df_reindex.loc['bar'])
'''
        k         v
second             
one     0 -0.141858
two     0 -0.786239
'''
# print(df_reindex.loc(axis=0)[:, 'two'])
'''
              k         v
first second             
bar   two     0 -0.615808
foo   two     0  0.238987
'''

# print(df_reindex.loc[('bar', 'two'):('foo', 'one')])
'''
              k         v
first second             
bar   two     0  1.055614
foo   one     0  0.899269
'''
# 交叉选择 xs(key, axis, level, drop_level)
# print(df_reindex.xs(key='two', level='second', drop_level=False))
'''
              k         v
first second             
bar   two     0 -0.615808
foo   two     0  0.238987
'''

# 多级列索引
df = df_reindex.T
# print(df)
'''
first        bar                 foo          
second       one       two       one       two
k       0.000000  0.000000  0.000000  0.000000
v      -0.062292  0.342238 -0.520897 -0.565263
'''
# print(df.loc(axis=1)[:, 'two'])
# print(df.xs('two', level=-1, axis=1, drop_level=False))
'''
first        bar       foo
second       two       two
k       0.000000  0.000000
v      -0.711593  1.917493
'''
# print(df.xs(('bar', 'two'), level=(0, -1), axis=1))
'''
first       bar
second      two
k       0.00000
v       0.38138
'''

# 交换索引层级 swaplevel()
# print(df.swaplevel(axis=1))
'''
second       one       two       one       two
first        bar       bar       foo       foo
k       0.000000  0.000000  0.000000  0.000000
v       0.398076 -0.346376  0.555577 -1.454103
'''

# 重命名索引
# print(df_reindex.rename(index={'bar': "bar_x", 'one': "one_x"}))
'''
              k         v
first second             
bar_x one_x   0  0.686636
      two     0  0.352534
foo   one_x   0  0.489657
      two     0  0.067204
'''
# 重命名轴名称
# print(df_reindex.rename_axis(index=['first_x', 'second_x']))
'''
                  k         v
first_x second_x             
bar     one       0  0.792857
        two       0 -1.092279
foo     one       0 -1.822016
        two       0  0.661191
'''

# B、时间序列与日期

# 注意：Pandas用NaT表示日期时间、时间差及时间段的空值，类似NaN

# 1）时间戳 pd.Timestamp()
# 日期计算：支持多种字符串类型如：20230501、2023-5-1、2023-05-01、2023/5/1、2023/05/01等
day = pd.Timestamp('20230501')
day = pd.Timestamp(2023, 5, 1)    # 类似datetime.datetime()
# 星期
print(day.day_name())     # Monday
# 月
print(day.month_name())   # May
# 获取年份、月份、天
print(day.year)           # 2023
print(day.month)          # 5
print(day.day)            # 1

# 其他Timestamp可以访问的日期/时间属性成员
'''
属性                        描述
year                       返回datetime的年
month                      返回datetime的月
day                        返回datetime的日
hour                       返回datetime的小时
minute                     返回datetime的分钟
second                     返回datetime的秒
date                       返回datetime.date（不包含时区信息）
time                       返回datetime.time（不包含时区信息）
timetz                     返回带本地时区信息的datetime.time
dayofyear                  一年里的第几天
week/weekofyear            一年里的第几周
weekday/dayofweek          一周里的第几天（Monday=0，Sunday=6）
quarter                    返回该天所处的季节（Jan-Mar=1，Apr-Jun=2等）
days_in_month              返回该天所在月有多少天
is_month_start             是否月初（由频率定义，下同）
is_month_end               是否月末
is_quarter_start           是否季初
is_quarter_end             是否季末
is_year_start              是否年初
is_year_end                是否年末
is_leap_year               是否闰年
'''
ts = pd.Timestamp('2023-05-01 08:00:00', tz='Asia/Shanghai')
print(ts.timetz())         # 08:00:00
print(ts.date())           # 2023-05-01
print(ts.is_month_start)   # True

# 2）时间段 pd.Period()
# freq: Y显示到年 M显示到月 D显示到天
# 仅支持yyyy-MM-dd hh:mm:ss格式
print(pd.Period('2023-05-20', freq='M'))    # 2023-05
print(pd.Period('2023-05'))                 # 2023-05
print(pd.Period('2023-05', freq='D'))       # 2023-05-01

# 3）转换时间戳 to_datetime(str,format,dayfirst,unit)
# 用于转换字符串、纪元式及混合日期时间的Series或列表
print(pd.to_datetime('Jul 10, 2023'))       # 2023-07-10 00:00:00
print(pd.to_datetime('2023/07/10'))         # 2023-07-10 00:00:00
print(pd.to_datetime('2023.07.10'))         # 2023-07-10 00:00:00
# 解析欧式日期（日-月-年）：使用参数：dayfirst
print(pd.to_datetime('01-07-2023', dayfirst=True))    # 2023-07-01 00:00:00
# 其他类型解析
print(pd.to_datetime('Sat, Apr, 15 2023 23:01:52'))   # 2023-04-15 23:01:52
print(pd.to_datetime('5/20/2023'))                    # 2023-05-20 00:00:00
print(pd.to_datetime('15 Oct, 2023 8:10am'))          # 2023-10-15 08:10:00
# 纪元(unix)时间戳转换 unit：单位（s或ms）
print(pd.to_datetime(1694593035, unit='s'))  # 2023-09-13 08:17:15
# format：仅用于加快转换速度
print(pd.to_datetime('2023-07-10', format='%Y-%m-%d'))

# 4）多列组合日期时间
df = pd.DataFrame({'year': [2022, 2023], 'month': [5, 6], 'day': [9, 10]})
print(df.to_string())
print(pd.to_datetime(df[['year', 'month', 'day']]))
'''
0   2022-05-09
1   2023-06-10
dtype: datetime64[ns]
'''

# 5）偏移日期时间：pd.DateOffset()
# DateOffset对象用于计算日历日时间偏移日期时间，类似dateutil.relativedelta
# 对应日历日时间
print(ts + pd.DateOffset(days=1))      # 2023-05-02 08:00:00+08:00
# 对应绝对时间
print(ts + pd.Timedelta(days=1))       # 2023-05-02 08:00:00+08:00
# 参数偏移
print(ts + pd.offsets.Week())          # 2023-05-08 08:00:00+08:00

# 6）时区处理、时区转换
# 1）时区处理
# 利用pytz与datetuil或标准库datetime.timezone对象，Pandas能以多种方式处理不同时区的时间戳
'''
pytz：tz='Asia/Shanghai'
dateutil：tz=dateutil.tz.tzutc()
datetime：tz=datetime.timezone.utc
'''
import dateutil
import datetime

dft = pd.DataFrame({'dt': pd.date_range('20221001', periods=5, freq='MS', tz=datetime.timezone.utc), 'v': np.random.randn(5).round(2)*10})
print(dft.to_string())
'''
                         dt     v
0 2022-10-01 00:00:00+00:00   6.4
1 2022-11-01 00:00:00+00:00   1.0
2 2022-12-01 00:00:00+00:00   5.1
3 2023-01-01 00:00:00+00:00   2.6
4 2023-02-01 00:00:00+00:00  25.3
'''
# 2）时区转换
# 方式1：tz_convert()
# 注意：DataFrame需要将日期时间列设置为索引，Series需要设置日期时间索引
print(dft.set_index('dt').tz_convert('US/Eastern').reset_index())
'''
                         dt     v
0 2022-09-30 20:00:00-04:00   6.4
1 2022-10-31 20:00:00-04:00   1.0
2 2022-11-30 19:00:00-05:00   5.1
3 2022-12-31 19:00:00-05:00   2.6
4 2023-01-31 19:00:00-05:00  25.3
'''
# 方式2：astype()
dft['dt'] = dft['dt'].astype('datetime64[ns, Asia/Shanghai]')
print(dft.to_string())
'''
                         dt     v
0 2022-10-01 08:00:00+08:00   6.4
1 2022-11-01 08:00:00+08:00   1.0
2 2022-12-01 08:00:00+08:00   5.1
3 2023-01-01 08:00:00+08:00   2.6
4 2023-02-01 08:00:00+08:00  25.3
'''


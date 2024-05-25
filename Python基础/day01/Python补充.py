
import array
import numpy as np
import pandas as pd

# Python补充

# Python数组与列表的区别

# Python数组
# 创建数组：Python内置的array模块用于创建Python数组（ArrayType）
# 1）array.array()：仅支持一维数组
# 第一个参数表示类型，用于规定数组元素的类型
# 例如，i：int(大写为无符号int)，l：long(大写为无符号long)，f：float，d：double
arr1 = array.array('i', [2, 3, 5])
print(arr1.tolist())       # [2, 3, 5]
print(type(arr1))          # <class 'array.array'>
# 2）numpy.array()：支持多维数组
arr2 = np.array([2, 3, 5])
print(arr2.tolist())       # [2, 3, 5]
print(type(arr2))          # <class 'numpy.ndarray'>

# Python列表
ls = [0, 'abc', True]
print(ls)                  # [0, 'abc', True]
print(type(ls))            # <class 'list'>

# Python数组和列表反转
# Python切片格式：[sta:end:step]
# 当step>0时，sta缺省的为0，end缺省的为len-1，step缺省的为1，可简写为[::]
# 当step<0时，sta缺省的为-1，end缺省的为-(len-1)，step缺省的为-1，可简写为[::-1]，表示倒序
# 数组反转
print(arr1[::-1].tolist())   # [5, 3, 2]
print(arr1.reverse())        # None
# 列表反转
print(ls[::-1])              # [True, 'abc', 0]
print(ls.reverse())          # None

# 值得注意的是，Python数组和列表对象的reverse()方法反转结果都为None，官方也没提示什么原因

# Python数组的其他操作
# 一次添加单个元素
arr1.append(7)
print(arr1.tolist())         # [5, 3, 2, 7]
# 一次添加多个元素
arr1.extend([11, 13])
print(arr1.tolist())         # [5, 3, 2, 7, 11, 13]

# 修改元素
arr1[0] = 1
print(arr1.tolist())         # [1, 3, 2, 7, 11, 13]

# 删除元素
arr1.remove(3)
print(arr1.tolist())         # [1, 2, 7, 11, 13]
# 根据索引删除
arr1.pop(3)
print(arr1.tolist())         # [1, 2, 7, 13]

# 总结：
# 1）Python数组和列表具有相同的数据存储方式。数组只能包含一种数据类型的元素，列表可以包含任何数据类型的元素
# 2）Python数组和列表都是有序、可变长度、值可重复的，数组需要使用array模块声明，列表不用
# 3）Python数组和列表都支持索引、切片、迭代等
# 4）存储相同数量的数据，数组使用较紧凑的内存连续的方式存储数据，占用内存较小，而列表消耗内存较多
# 5）数组更适合数据运算，数组一般使用np.array()，不用array模块，列表比数组更加灵活，如无特殊需要，不建议使用数组


# split()、rsplit()、splitlines()字符串分割

# split(sep,maxsplit=-1)
# - sep：分割符
# - maxsplit：从左向右找到第一个（分割次数）匹配的分割符进行分割，默认-1，按全部分割符分割
# 如果不指定分割符，split()将以空白符作为分割符
s = 'path/a/b/c'
print(s.split('/'))                 # ['path', 'a', 'b', 'c']
print(s.split('/', maxsplit=1))     # ['path', 'a/b/c']

# rsplit(sep,maxsplit=-1)
# - sep：分割符
# - maxsplit：从右向左找到第一个（分割次数）匹配的分割符进行分割，默认-1，按全部分割符分割
# 如果不指定分割符，rsplit()将以空白符作为分割符
print(s.rsplit('/', maxsplit=1))    # ['path/a/b', 'c']

# splitlines()
# 根据换行符\r（回车）、\r\n（回车并换行）、\n（换行）进行分割
s = 'path\ra\r\nb\nc'
print(s.splitlines())               # ['path', 'a', 'b', 'c']


# 日期操作相关
# 日期转化为天数

import time
import datetime
from dateutil import parser
import pandas as pd

dt1 = '2024-01-11'  # 字符串可转换为datetime类型处理

dt2 = time.localtime()
print(type(dt2))  # <class 'time.struct_time'>
print(dt2)        # time.struct_time(tm_year=2024, tm_mon=1, tm_mday=11, tm_hour=16, tm_min=27, tm_sec=18, tm_wday=3, tm_yday=11, tm_isdst=0)

dt3 = datetime.datetime.now()
print(type(dt3))  # <class 'datetime.datetime'>
print(dt3)        # 2024-01-11 16:27:18.659806

dt4 = datetime.date.today()
print(type(dt4))  # <class 'datetime.date'>
print(dt4)        # 2024-01-11

# 获取年月日

# 字符串类型转换为datetime类型
print(pd.to_datetime(dt1))                          # 2024-01-11 00:00:00
print(parser.parse(dt1))                            # 2024-01-11 00:00:00
print(datetime.datetime.strptime(dt1, '%Y-%m-%d'))  # 2024-01-11 00:00:00

# struct_time类型获取
print(dt2.tm_mon)                 # 1
y2, m2, d2 = dt2[:3]
print(y2, m2, d2)                 # 2024 1 11

# datetime类型获取
print(dt3.month)                  # 1
y3, m3, d3 = dt3.timetuple()[:3]  # timetuple()将datetime类型转换为struct_time类型
print(y3, m3, d3)                 # 2024 1 11

# date类型获取
print(dt4.month)                  # 1
y4, m4, d4 = dt4.timetuple()[:3]  # 同上
print(y4, m4, d4)                 # 2024 1 11

import calendar

# 判断是否是闰年
print(calendar.isleap(y2))  # True
# 获取指定年指定月的天数
# monthrange()返回两个值，第一个为该月第1天是周几（0~6表示），第二个为该月天数
print(calendar.monthrange(2024, 2)[1])  # 29

# 获取一年中的第几天
# 方式1
# 获取指定年所有月的天数列表
months_days = [calendar.monthrange(2024, m + 1)[1] for m in range(12)]
print(months_days)  # [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

print(sum(months_days[:m2 - 1]) + d2)  # 11
print(sum(months_days[:m3 - 1]) + d3)  # 11
print(sum(months_days[:m4 - 1]) + d4)  # 11

# 方式2
# struct_time类型获取：使用tm_yday属性
print(dt2.tm_yday)              # 11
# datetime类型获取
print(dt3.timetuple().tm_yday)  # 11
# date类型获取
print(dt4.timetuple().tm_yday)  # 11

# 方式3
print((pd.to_datetime(dt1) - datetime.datetime(pd.to_datetime(dt1).year, 1, 1)).days + 1)


# 季度初末日期区间

# 获取指定年份季度范围
def get_quarter_range(s: int | str, fmt='%Y%m%d'):
    # 季度起始日期
    quarter_sta_dts = [pd.to_datetime(f'{s}-{m}-01').date().strftime(fmt) for m in [1, 4, 7, 10]]
    # 季度末日期
    month_days = [calendar.monthrange(int(s), m + 1)[1] for m in range(12)]
    quarter_end_dts = [pd.to_datetime(f'{s}-{idx + 1}-{days}').date().strftime(fmt) for idx, days in enumerate(month_days) if (idx + 1) % 3 == 0]
    # 组合
    quarter_range = list(zip(quarter_sta_dts, quarter_end_dts))
    quarter_range_dict = [{f'{s}Q{idx + 1}': ran} for idx, ran in enumerate(quarter_range)]
    return quarter_range_dict

print(get_quarter_range(2024))
'''
[{'2024Q1': ('20240101', '20240331')}, {'2024Q2': ('20240401', '20240630')}, {'2024Q3': ('20240701', '20240930')}, {'2024Q4': ('20241001', '20241231')}]
'''


# Python中的字符串编码和解码

# 和其他语言不同，Python3中有3种类型的字符串对象：str、unicode和bytes，它们之间的转换关系可以通过编码（encode）和解码（decode）描述
# 例如，Requests模块中的`resp.text`响应的是`unicode`类型的字符串，`resp.content`则响应的是bytes类型的字符串，这时候就需要转换

# str.encode(encoding)：字符串（str）=> 二进制（bytes）/unicode
# bytes.decode(encoding)：二进制（bytes）/unicode => 字符串（str）

# 3种字符串类型对应的转换关系如下：

# 字符串 => 二进制（ASCII）
s1 = "中文"
s2 = s1.encode()
print(s2)         # b'\xe4\xb8\xad\xe6\x96\x87'
print(type(s2))   # <class 'bytes'>

# 字符串 => unicode
# 参数encoding取值：unicode-escape或raw_unicode_escape
s3 = s1.encode('unicode-escape')
print(s3)         # b'\\u4e2d\\u6587'
print(type(s3))   # <class 'bytes'>

# 二进制（ASCII） => 字符串
s4 = s2.decode()
print(s4)        # 中文
print(type(s4))  # <class 'str'>

# unicode => 字符串
# 参数encoding取值：unicode-escape或raw_unicode_escape
s5 = s3.decode('unicode-escape')
print(s5)        # 中文
print(type(s5))  # <class 'str'>




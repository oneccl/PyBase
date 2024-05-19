
# loc[]与iloc[]区别
# map()、apply()、applymap()/df.map()区别

import numpy as np
import pandas as pd

df = pd.DataFrame({'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]})
print(df)
'''
   A  B  C
0  1  2  3
1  4  5  6
2  7  8  9
'''

# A、loc[]与iloc[]区别

# 1）选取第1行数据
print(list(df.loc[0]))
print(list(df.iloc[0]))
'''
[1, 2, 3]
'''
# 结论1：选取指定行数据
# loc[]和iloc[]在行参数上都使用行索引，行索引从0开始

# 2）选取1-2行数据
print(df.loc[0: 1])
print(df.iloc[0: 2])
'''
   A  B  C
0  1  2  3
1  4  5  6
'''
# 结论2：选取多行
# loc[]：左闭右闭[sta, end]
# iloc[]：左闭右开[sta, end)

# 3）选取指定位置值
print(df.loc[1, 'C'])
print(df.iloc[1, 2])
'''
6
'''
# 结论3：选取指定位置值
# loc[row_index, col_name]：loc[]在行参数上使用行索引，在列参数上必须使用列字段名，df.loc[1, 2]会报错：KeyError
# iloc[row_index, col_index]：loc[]在行参数上使用行索引，在列参数上必须使用列索引，df.iloc[1, 'C']会报错：ValueError

# 4）选取第一列数据
print(list(df.loc[:, 'A']))
print(list(df.iloc[:, 0]))
'''
[1, 4, 7]
'''
# 结论4：选取指定列数据
# loc[row_index(切片), col_name(切片)]：loc[]在列参数上必须使用列字段名，df.loc[:, 0]会报错：KeyError
# iloc[row_index(切片), col_index(切片)]：loc[]在列参数上必须使用列索引，df.iloc[:, 'A']会报错：ValueError

# 5）选取多行多列
print(df.loc[0: 1, 'A': 'B'])
print(df.iloc[0: 2, 0: 2])
'''
   A  B
0  1  2
1  4  5
'''
# 结论5：多行多列切片
# loc[row_index(切片), col_name(切片)]：行列切片左闭右闭[sta, end]，在列切片参数上必须使用列字段名，df.loc[0: 1, 0: 1]会报错：TypeError
# iloc[row_index(切片), col_index(切片)]：行列切片左闭右开[sta, end)，在列切片参数上必须使用列索引，df.iloc[0: 2, 'A': 'C']会报错：TypeError

# 6）选取指定多行多列
print(df.loc[[0, 1], ['A', 'C']])
print(df.iloc[[0, 1], [0, 2]])
'''
   A  C
0  1  3
1  4  6
'''
# 结论6：选取指定多行多列
# loc[row_index_list, col_name_list]：loc[]在列参数上必须使用列字段名列表，df.loc[[0, 1], [0, 2]]会报错：KeyError
# iloc[row_index_list, col_index_list]：iloc[]在列参数上必须使用列索引列表，df.iloc[[0, 1], ['A', 'C']]会报错：IndexError

# 7）loc[]布尔索引
# 选择满足筛选条件的子df的指定列，返回DataFrame或Series类型
res1 = df.loc[df['A'] % 2 != 0, ['B', 'C']]
print(res1)
'''
   B  C
0  2  3
2  8  9
'''
res2 = df.loc[(df['A'] > 3) & (df['C'] < 7), ['B', 'C']]
print(res2)
'''
   B  C
1  5  6
'''
# 行列级别布尔选择
# 单行布尔筛选
# 筛选指定行满足条件的值
print(list(filter(lambda x: x % 2 != 0, df.loc[0])))    # [1, 3]

# 单列布尔筛选
# 筛选B列满足条件的A列值
print(df[df['B'] > 3]['A'].tolist())        # [4, 7]
print(df.query('B > 3')['A'].tolist())      # [4, 7]
print(df['A'].loc[df['B'] > 3].tolist())    # [4, 7]

# 结论7：
# loc[bool表达式, cols]：支持布尔索引，可以筛选满足条件的子df的指定列
# iloc[]：不支持布尔索引，报错：NotImplementedError: iLocation based boolean indexing on an integer type is not available

# 总结：
# 1）loc[]和iloc[]在行参数上都是使用行索引，且行索引从0开始
# 2）在行列参数上，loc[]范围始终为左闭右闭[sta, end]，iloc[]范围始终为左闭右开[sta, end)
# 3）只要有列参数参与，无论什么形式，loc[]在列参数上必须使用列字段名，iloc[]在列参数上必须使用列索引，且索引从0开始
# 4）表达式：loc[row_index(切片/列表), column_name(切片/列表)]，iloc[row_index(切片/列表), column_index(切片/列表)]
# 5） loc[bool表达式, cols]支持布尔索引，iloc[]不支持布尔索引


# B、map()、apply()、applymap()/df.map()区别

# 1）map(func, *iter)：Python内置高级函数，用于对一个或多个可迭代序列的每个元素执行函数func
df['D'] = list(map(lambda a, b: a + b, df.A, df.B))
print(df)
'''
   A  B  C   D
0  1  2  3   3
1  4  5  6   9
2  7  8  9  15
'''
df['E'] = list(map(lambda a, b: [a, b], df.A, df.B))
print(df)
'''
   A  B  C   D       E
0  1  2  3   3  [1, 2]
1  4  5  6   9  [4, 5]
2  7  8  9  15  [7, 8]
'''
# 应用多个函数收集：对A、B两列应用求平均和求和函数，将结果收集到列表
df['F'] = [list(map(lambda f: f(a, b), [lambda x, y: (x+y)/2, lambda x, y: x+y])) for a, b in zip(df.A, df.B)]
print(df)
'''
   A  B  C   D       E          F
0  1  2  3   3  [1, 2]   [1.5, 3]
1  4  5  6   9  [4, 5]   [4.5, 9]
2  7  8  9  15  [7, 8]  [7.5, 15]
'''

# 2）apply(func, axis=0)：Pandas DataFrame/Series对象方法，对df某一列/多列或某一行/多行中的元素执行函数func，默认axis=0为列
# 对某一列的每个元素执行函数
df['D'] = df['A'].apply(lambda e: e ** 2)
# 对指定多列的每个元素执行函数
df[['E', 'F']] = df[['A', 'B']].apply(lambda e: -e)
# 对某一行的每个元素执行函数
# 除了pd.concat()，在原df基础上增量追加一行可使用如下方式（不能使用该方式添加多行）
df.loc[3] = df.loc[0].apply(lambda e: e ** 2)
# 对指定多行的每个元素执行函数
df = pd.concat([df, df.loc[0: 1, :].apply(lambda e: -e)], ignore_index=True)
print(df)
'''
   A  B  C   D  E  F
0  1  2  3   1 -1 -2
1  4  5  6  16 -4 -5
2  7  8  9  49 -7 -8
3  1  4  9   1  1  4
4 -1 -2 -3  -1  1  2
5 -4 -5 -6 -16  4  5
'''

# 对所有列分别计算sum
# df.index.values[-1]：获取最后一个索引值
df.loc[df.index.values[-1]+1] = df.apply(np.sum)
print(df)
'''
    A   B   C
0   1   2   3
1   4   5   6
2   7   8   9
3  12  15  18
'''
# 对所有行分别计算sum
df['sum'] = df.apply(np.sum, axis=1)
print(df)
'''
   A  B  C  sum
0  1  2  3    6
1  4  5  6   15
2  7  8  9   24
'''
# 对DataFrame每个元素执行函数
print(df.apply(lambda e: e + 1))
'''
   A  B   C
0  2  3   4
1  5  6   7
2  8  9  10
'''

# 3）applymap(func)/df.map(func)：Pandas DataFrame对象方法，Series不支持，用于对df的每个元素执行函数func，且将来会被df.map()替换
# AttributeError: 'Series' object has no attribute 'applymap'
# FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
# 对DataFrame每个元素执行函数
print(df.applymap(lambda e: e + 1))
print(df.map(lambda e: e + 1))
'''
   A  B   C
0  2  3   4
1  5  6   7
2  8  9  10
'''

# 总结：
# 1）map(func, *iter)是Python内置高级函数，可以直接调用，用于对一个或多个可迭代序列的每个元素执行函数
# 2）apply(func, axis=0)是DataFrame/Series对象的方法，用于对df某一列/多列或某一行/多行中的每个元素执行函数
# 3）applymap(func)仅是DataFrame对象的方法，Series不支持，不能指定轴axis，用于对df的每个元素执行函数，且将来会被df.map()替换
# 4）在对DataFrame或Series运用apply()、applymap()的时候，必须保证所有的字段类型与函数的参数一致


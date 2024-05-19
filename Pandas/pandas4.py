"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/9
Time: 13:33
Description:
"""

# Pandas复杂类型解析（常见问题及解决）

import numpy as np
import pandas as pd

# 1）列存在空值转换格式问题

df = pd.DataFrame({'int1': [np.nan, 2, 3], 'int2': [1, 2, None]})
print(df.to_string())
# 数据中的空值会自动转换成np.NaN
'''
   int1  int2
0   NaN   1.0
1   2.0   2.0
2   3.0   NaN
'''
# np.NaN的类型是np.float64格式，int类型转换为float类型不会丢失精度
# 列存在空值读取结果为float64（x.0）问题解决：将该列转换为int64类型
# 列存在空值转换为int64报错问题解决：将int64换为Int64
df['int1'] = df['int1'].astype('Int64')
df['int2'] = df['int2'].astype('Int64')
print(df.to_string())
'''
   int1  int2
0  <NA>     1
1     2     2
2     3  <NA>
'''

# 2）科学计数法类型转换精度丢失问题

# 解决：不要使用Excel打开CSV文件（会丢失精度）
df = pd.DataFrame({'id1': [np.nan, 20600000002083], 'id2': [1000600000000146, None]})
print(df.to_string())
'''
            id1           id2
0           NaN  1.000600e+15
1  2.060000e+13           NaN
'''
df[['id1', 'id2']] = df[['id1', 'id2']].astype('Int64').astype('string')
print(df.to_string())
'''
              id1               id2
0            <NA>  1000600000000146
1  20600000002083              <NA>
'''

# 3）列表样式的字符串处理

df = pd.DataFrame({"c1": ["['AB', 'BC']", "['CD', 'DE']"], "c2": [['AB', 'BC'], ['CD', 'DE']]})
print(df.to_string())
'''
             c1        c2
0  ['AB', 'BC']  [AB, BC]
1  ['CD', 'DE']  [CD, DE]
'''
# 对于c2（字符串列表），无需处理
# 对于c1（列表样式的字符串），需要进行类型转换

df['c1'] = df['c1'].apply(lambda x: ast.literal_eval(x) if x != '' else [])
print(df.to_string())
'''
         c1        c2
0  [AB, BC]  [AB, BC]
1  [CD, DE]  [CD, DE]
'''

# 4）对象列表处理

df = pd.DataFrame({
    'c1': ["[{'name': 'A', 'age': None}, {'name': 'B', 'age': 18}]", ""],
    'c2': [[{'name': 'A', 'age': None}, {'name': 'B', 'age': 20}], None]
})
print(df.to_string())
'''
                                                       c1                                                      c2
0  [{'name': 'A', 'age': None}, {'name': 'B', 'age': 18}]  [{'name': 'A', 'age': None}, {'name': 'B', 'age': 20}]
1                                                                                                            None
'''
# 对于c2（对象列表），无需处理
# 对于c1（对象列表样式的字符串），需要使用json模块进行格式转换
import json
# Json字符串中的属性(键)为单引号时解析报错：将单引号替换为双引号
# Json字符串中存在为None的值时解析报错：将None替换为null
json_str = "[{'name': 'A', 'age': None}, {'name': 'B', 'age': 18}]"
json_obj = json.loads(json_str.replace("'", '"').replace('None', 'null'))
print(json_obj)
print(type(json_obj))
print(json.dumps(json_obj))
'''
[{'name': 'A', 'age': None}, {'name': 'B', 'age': 18}]
<class 'list'>
[{"name": "A", "age": null}, {"name": "B", "age": 18}]
'''
text = df['c1'].values[0]
# json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置ensure_ascii=False
res = [json.dumps(j, ensure_ascii=False) for j in json.loads(text.replace("'", '"').replace('None', 'null'))]
print(res)
print(type(res))
'''
['{"name": "A", "age": null}', '{"name": "B", "age": 18}']
<class 'list'>
'''
# json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置ensure_ascii=False
df['c1_p'] = df['c1'].apply(lambda x: [json.dumps(j, ensure_ascii=False) for j in json.loads(x.replace("'", '"').replace('None', 'null'))] if x != "" else [])
print(df.to_string())
'''
                                                       c1                                                      c2                                                    c1_p
0  [{'name': 'A', 'age': None}, {'name': 'B', 'age': 18}]  [{'name': 'A', 'age': None}, {'name': 'B', 'age': 20}]  [{"name": "A", "age": null}, {"name": "B", "age": 18}]
1                                                                                                            None                                                      []
'''

# 5）对象处理

df = pd.DataFrame({
    'c1': ["{'name': 'A', 'age': None}", "{'name': 'B', 'age': 18}", ""],
    'c2': [{'name': 'A', 'age': None}, {'name': 'B', 'age': 20}, None]
})
print(df.to_string())
'''
                           c1                          c2
0  {'name': 'A', 'age': None}  {'name': 'A', 'age': None}
1    {'name': 'B', 'age': 18}    {'name': 'B', 'age': 20}
2                                                    None
'''
# 对于c2（json对象），无需处理
# 对于c1（json样式的字符串），需要使用json模块进行格式转换
df['c1_p'] = df['c1'].apply(lambda x: json.loads(x.replace("'", '"').replace('None', 'null')) if x != "" else {})
print(df.to_string())
'''
                           c1                          c2                        c1_p
0  {'name': 'A', 'age': None}  {'name': 'A', 'age': None}  {'name': 'A', 'age': None}
1    {'name': 'B', 'age': 18}    {'name': 'B', 'age': 20}    {'name': 'B', 'age': 18}
2                                                    None                          {}
'''

# 6）复杂对象处理

df = pd.DataFrame({
    'c1': ["{'name': 'A', 'age': None, 'addr': '53'}", "{'friends': [1,2,3]}", ""]
})
print(df.to_string())
'''
                                         c1
0  {'name': 'A', 'age': None, 'addr': '53'}
1                      {'friends': [1,2,3]}
2                                          
'''
# 对于c1（复杂json样式的字符串），需要使用json模块进行格式转换
# json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置ensure_ascii=False
df['c1_p'] = df['c1'].apply(lambda x: [json.dumps(eval(x), ensure_ascii=False)][0] if x != "" else np.nan)
print(df.to_string())
'''
                                         c1                                      c1_p
0  {'name': 'A', 'age': None, 'addr': '53'}  {"name": "A", "age": null, "addr": "53"}
1                      {'friends': [1,2,3]}                    {"friends": [1, 2, 3]}
2                                                                                 NaN
'''

# 补充：eval()函数与ast.literal_eval()

# 1、eval(expr)：Python内置函数：用于返回传入字符串的表达式结果

# 1）将字符串转化为表达式求结果
print(eval("2+3"))    # 5
# 2）将字符串转化为其它数据类型，如字典、列表、元组、集合等
# 将字符串转换为字典
print(eval("{'name': 'Tom', 'age': 18}"))    # {'name': 'Tom', 'age': 18}
# 将字符串转换为列表
ls = eval("[[1, 2], 10, {'k': 'v'}, {'arr': [{'k1': 'v1'}, '']}]")
print(ls)     # [[1, 2], 10, {'k': 'v'}, {'arr': [{'k1': 'v1'}, '']}]
for i in ls:
    print(i, type(i))
'''
[1, 2] <class 'list'>
10 <class 'int'>
{'k': 'v'} <class 'dict'>
{'arr': [{'k1': 'v1'}, '']} <class 'dict'>
'''
print(ls[3]['arr'])               # [{'k1': 'v1'}, '']
print(type(ls[3]['arr']))         # <class 'list'>
print(ls[3]['arr'][0]['k1'])      # v1
# 3）执行命令：将字符串转换为表达式执行
# eval()函数存在安全漏洞，类似SQL注入
str_cmd = "__import__('os').getcwd()"
print(eval(str_cmd))

# 2、ast.literal_eval()代替eval()实现数据类型转换

# literal_eval()会判断内容经过运算后是否为合法的Python类型，如果是则进行运算，否则不进行运算
import ast

print(ast.literal_eval("['AB', 'BC']"))     # ['AB', 'BC']
print(ast.literal_eval("{'name': 'Tom', 'age': 18}"))    # {'name': 'Tom', 'age': 18}
ls = ast.literal_eval("[[1, 2], 10, {'k': 'v'}, {'arr': [{'k1': 'v1'}, '']}]")
print(ls)     # [[1, 2], 10, {'k': 'v'}, {'arr': [{'k1': 'v1'}, '']}]


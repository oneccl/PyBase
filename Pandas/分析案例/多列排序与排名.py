
import numpy as np
import pandas as pd
import glob
import os

# Pandas多列排序与多列排名

files = glob.glob(os.path.join(r'C:\Users\cc\Desktop\Test', '*.xlsx'))

# 读取并合并Excel
df_res = pd.DataFrame()
for file in files:
    # 读取
    df = pd.read_excel(file)
    # 合并
    df_res = pd.concat([df_res, df], ignore_index=True)

# print(df_res.to_string())

# 实验需求：将每个人的积分、评分汇总，并按总积分排名，总积分一致时，按总评分排名，最终结果按排名升序

# 计算总积分总评分
df = df_res.\
    groupby('姓名', as_index=False).agg({'月度积分': 'sum', '月度评分': 'sum'}).\
    rename(columns={'月度积分': '总积分', '月度评分': '总评分'})

# 指定姓名、总积分列去重，保留第一个
df.drop_duplicates(subset=['姓名', '总积分'], keep='first', inplace=True)
print(df.to_string())
'''
   姓名  总积分  总评分
0  张三   18    32
1  李四   18    36
2  王五   19    38
3  赵六   16    35
'''

# 方式1：按总积分进行排序，若总积分相同则按照总评分排序（dense_rank()）

# # 按总积分排名
# df['总排名'] = df['总积分'].rank(method='dense', ascending=False)
# print(df.to_string())
# '''
#    姓名  总积分  总评分  总排名
# 0  张三   18    32     2.0
# 1  李四   18    36     2.0
# 2  王五   19    38     1.0
# 3  赵六   16    35     3.0
# '''
# # 按总积分进行排序，若总积分相同则按照总评分排序
# df.sort_values(by=['总积分', '总评分'], ascending=[False, False], inplace=True)
# print(df.to_string())
# '''
#    姓名  总积分  总评分  总排名
# 2  王五   19    38     1.0
# 1  李四   18    36     2.0
# 0  张三   18    32     2.0
# 3  赵六   16    35     3.0
# '''

# 方式2：按总积分进行排名，若总积分相同则按照总评分排名（row_number()）
# 使用辅助列
df['辅助列'] = df.eval('总积分*100 + 总评分')
# print(df.to_string())
df['总排名'] = df['辅助列'].rank(method='dense', ascending=False)
df.sort_values(by='总排名', ignore_index=True, ascending=True, inplace=True)
# 删除辅助列
df.drop(axis=1, columns='辅助列', inplace=True)
print(df.to_string())
'''
   姓名  总积分  总评分  总排名
0  王五   19    38     1.0
1  李四   18    36     2.0
2  张三   18    32     3.0
3  赵六   16    35     4.0
'''

# 补充：rank()的pct参数使用
# pct：返回相对排名（每个值在数据中的位置的百分比），百分比表示每个元素在数据集中的相对位置，默认False

# 按照总评分计算相对排名
# 例如一个相对排名为0.75的人，他的评分高于75%的人
df['相对排名'] = df['总评分'].rank(pct=True)
print(df.to_string())
'''
   姓名  总积分  总评分  总排名  相对排名
0  王五   19    38     1.0    1.00
1  李四   18    36     2.0    0.75
2  张三   18    32     3.0    0.25
3  赵六   16    35     4.0    0.50
'''
# 获取相对排名前50%的人
print(df.query('相对排名 > 0.50'))
'''
   姓名  总积分  总评分  总排名  相对排名
0  王五   19    38     1.0    1.00
1  李四   18    36     2.0    0.75
'''


"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/10/14
Time: 11:53
Description:
"""

# Python-SQLite3
# SQLite是基于文件的数据库
# SQLite3模块是Python内置的数据库模块，适用于小型应用和快速原型开发

import sqlite3

# 连接数据库
cn = sqlite3.connect('db_name')

# 获取游标
cursor = cn.cursor()

# 插入一条数据
cursor.execute('sql', 'param')
# 批量插入
cursor.executemany('sql', 'params')

# 执行SQL(如查询)
cursor.execute('sql')

# 获取一条记录
row = cursor.fetchone()
# 获取所有记录
rows = cursor.fetchall()

# 关闭游标和连接
cursor.close()
cn.close()


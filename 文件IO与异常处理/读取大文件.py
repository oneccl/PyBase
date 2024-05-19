"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2024/1/17
Time: 22:52
Description:
"""

# Python提供了多种方法来读取文件，例如，使用Python的标准文件读取流程，即使用open()函数打开一个文件，然后使用readline()或readlines()方法逐行读取文件内容

# 如果文件过大，这种方法可能会导致内存不足的问题，因为它需要将整个文件读入内存中

# 如果需要处理大文件，可以使用open()函数先打开一个文件，然后使用read()方法并指定块大小读取文件，这种方法可以有效的避免内存不足的问题

def chunked_file_reader(fp, block=1024*2):
    while True:
        chunked = fp.read(block)
        if not chunked:
            break
        yield chunked

def chunked_processor(file, block=1024*2):
    with open(file, 'r', encoding='utf-8') as fp:
        for chunked in chunked_file_reader(fp, block):
            # 处理块文件
            ...

# 补充：Python如何处理大型（>1G）CSV文件？

import time

file = 'file.csv'

# A、使用Pandas提供的API参数
# Pandas提供了一些方法参数可以解决这种问题，使得读取大型CSV文件变得更加容易
import numpy as np
import pandas as pd

# # 1）使用usecols加载部分列数据，避免全列读取；使用dtype转换类型读取数据，减少内存占用
# data = pd.read_csv(file, usecols=usecols, dtype=dtype)
# print(len(data))
# print(data.head().to_string())

# 2）分批分块读取
# 可以使用`chunksize`参数将数据分成多个块读取，每个数据块包含chunksize行数据，以免发生内存不足的问题
# # 每次读取行数为1000
# # pd.read_csv()返回类型：<class 'pandas.io.parsers.readers.TextFileReader'>
# for chunk in pd.read_csv(file, chunksize=1000, iterator=True):
#     print(type(chunk))    # <class 'pandas.core.frame.DataFrame'>
#     # 每次处理1000行数据
#     print(len(chunk))
#     print(chunk.head().to_string())

# # 3）使用C引擎
# # C引擎相较于默认的Python引擎更快
# start = time.perf_counter()
# data = pd.read_csv(file, dtype=str, engine="c")
# print(len(data))
# print(data.head().to_string())
# print(time.perf_counter() - start)   # 75.78744679992087
#
# # B、使用第三方库：Dask并行分布式计算Python库
# # 使用Dask库（Pandas官方推荐）
# import dask.dataframe as dd
#
# start = time.perf_counter()
# # 返回类型：<class 'dask_expr._collection.DataFrame'>
# data = dd.read_csv(file, dtype=str)
# print(len(data))
# print(data.head().to_string())
# print(time.perf_counter() - start)   # 52.351866899989545




# 数据处理基础库Numpy、Pandas
import numpy as np
import pandas as pd

# Pandas 数据分析库
"""
Pandas是一个用于数据分析和处理的强大库，它基于NumPy构建，并提供了更高级
的数据结构和分析工具。Pandas的两个核心数据结构是Series和DataFrame
"""

# 1、Pandas数据结构：Series和DataFrame
'''
1）Series：类似于一维数组，是一种带有标签的数据结构。它由一组数据和与之相关的索引组成
可以使用pd.Series()函数创建一个Series对象
2）DataFrame：类似于二维表格或电子表格，是Pandas中最常用的数据结构。它由一组Series对象
组成，每个Series对象代表表格中的一列。可以使用pd.DataFrame()函数创建一个DataFrame对象
'''
'''
1）Series
Series是类似表格的一个列（columns），类似于一维数组，可以存储任何数据类型
Series由索引和列组成，函数如下：pandas.Series(data,index)
'''
# 使用列表创建，默认行索引为0、1、2、...
s = pd.Series([1, 3, np.nan, 6, 8])
print(s)
'''
None：None表示空值，类型是NoneType
NaN（np.nan）：当使用numpy或pandas处理数据时，会将表中空缺项(为空)转换为NaN（类型是float）
'''
# 使用列表创建，指定行索引
arr = [2, 3,  5]
series = pd.Series(arr, index=["x", "y", "z"])
print(series)

# 通过字典类型创建Series，字典的Key为索引
dic = {1: "a", 2: "b", 3: "c"}
dic_series = pd.Series(dic)
print(dic_series)

# 根据索引值读取数据
print(series["y"])          # 3
print(dic_series[3])        # c

"""
2）DataFrame
DataFrame是一个二维表格型数据结构，类似二维数组
DataFrame由行索引、列索引和列组成，构造方法：pandas.DataFrame(data,index,columns)
"""
# 使用二维列表创建
data1 = [["a", 5], ["b", 10], ["c", 12]]
df1 = pd.DataFrame(data1)
print(df1)

# 使用ndarrays创建
data2 = {"Site": ["Tom", "Jack"], "Age": [18, 17]}
df2 = pd.DataFrame(data2)
print(df2)

# 使用字典创建，没有对应的部分数据为NaN
data3 = [{"a": 2, "b": 3}, {"a": 5, "b": 10, "c": 11}]
df3 = pd.DataFrame(data3)
print(df3)

# 使用Series字典对象创建
df = pd.DataFrame({
    'A': pd.Timestamp('20230502'),
    'B': pd.Series([1, 2, 3], index=list(range(3)), dtype='float64'),
    'C': np.array([4] * 3, dtype='int64')
})
print(df.to_string())
'''
           A    B  C
0 2023-05-02  1.0  4
1 2023-05-02  2.0  4
2 2023-05-02  3.0  4
'''

# ※ DataFrame的操作: 见本页下面

# pandas基础操作（文件读写）
'''
数据加载：可以使用pd.read_csv()函数从CSV文件中加载数据，或使用pd.read_excel()函数从Excel文件中加载数据
数据清洗：可以使用Pandas的函数和方法来处理缺失值、重复值、异常值等数据问题。例如，使用df.dropna()删除包含缺失值的行或列
数据整理：可以对数据进行排序、重塑、合并等操作，以满足分析和可视化的需求
数据分组和聚合：可以使用df.groupby()方法将数据按照某个或多个条件进行分组，并应用聚合函数进行计算
数据分析技巧：Pandas提供了丰富的数据分析工具，如数据透视表、时间序列分析、数据可视化等
'''
# 1）CSV文件：使用,分割值

# 读取CSV文件，返回前5行和末5行的数据，中间以...代替
# df4 = pd.read_csv("nba.csv")

# 将DataFrame数据存储为CSV文件
# df3.to_csv("data3.csv")

# CSV/文本文件 示例：
# 1.1）读取 read_csv()
# 常用参数解析：
"""
1）filepath_or_buffer：文件路径、url或具有方法的任何对象（如StringIO()）

2）sep：分隔符，默认','，可以使用正则表达式

3）header：表头，整型或整型列表类型，默认将第一行作为列名，数组可设置多级列；header会忽略空行和注释行，因此header=0表示第一行数据而非文件的第一行

4）names：列名，列表类型，与数据一一对应，若文件不包含列名，则应该设置header=None；若文件包含列名，重新命名列名后可以删除原来列所在的行；列名列表中不允许有重复值

5）index_col：索引，DataFrame的行索引列表，可以是列编号或列名，Pandas不再使用首列作为索引，会自动使用以0开始的自然索引
pd.read_csv(data, index_col=n)                 # 指定第几列作为索引
pd.read_csv(data, index_col='col')             # 指定列名列作为索引
pd.read_csv(data, index_col=[n1, ...])         # 指定哪些列作为多级索引
pd.read_csv(data, index_col=['col1', ...])     # 指定哪些列作为多级索引

6）usecols：读取部分列，list类型或callable类型
pd.read_csv(data, usecols=[0,2,1])             # 按列索引读取指定列，与顺序无关
pd.read_csv(data, usecols=['col1', 'col3'])    # 按列名读取指定列，列名必须存在
from io import StringIO
# StringIO模块是Python的标准库之一，它提供了一个类似于文件对象的接口，使用的是内存缓冲区做为数据源，而不是磁盘上实际存在的文件。可以通过类似文件的操作方式对数据进行操作，特别是在对字符串进行处理时非常方便
pd.read_csv(StringIO(data), usecols=lambda x: x not in ['col1', 'col2'])   # 筛选列

7）squeeze：返回Series还是DataFrame，默认False，若设置为True，如果文件只包含一列或仅读取一列，则返回一个Series，如果有多列，则还是返回DataFrame
pd.read_csv(data, usecols=[0], squeeze=True)   # 只读取一列，返回一个Series

8）prefix：表头前缀，若原始数据没有列名或重复列名，可以指定一个前缀加数字的名称作为列名，如N0、N1、...
pd.read_csv(data, prefix='N', header=None)

9）mangle_dupe_cols：重复列名处理，默认True，若为False，则当列名中有重复时，前列将会被后列覆盖

10）dtype：指定某些列或整体数据的数据类型
pd.read_csv(data, dtype=str)                   # 指定整体数据的数据类型
pd.read_csv(data, dtype={'col1': 'int64', 'col2': np.float64, ...})   # 指定某些列的数据类型
pd.read_csv(data, dtype=[str, datetime, ...])  # 依次指定所有列的数据类型

11）engine：引擎，分析引擎可以选择C或Python，C语言速度更快，Python语言功能最完善

12）converters：对列数据处理，字典参数中指定列名与对此列的处理函数，字典的键可以是列名或列的索引
pd.read_csv(StringIO(data), converters={'col': lambda/func, n: lambda/func})
pd.read_csv(StringIO(data), converters={'col': str})    # 转换类型

13）skipinitialspace：是否跳过分隔符后的空格，默认False

14）skiprows：跳过指定行，要跳过的行数（从文件开头算起）或要跳过的行号列表（从0开始）
pd.read_csv(data, skiprows=2)                  # 跳过前3行
pd.read_csv(data, skiprows=[2,3,5,...])        # 跳过指定行
pd.read_csv(StringIO(data), skiprows=lambda x: x % 2 != 0)     # 隔行跳过

skipfooter：尾部跳过，int类型
pd.read_csv(data, skipfooter=1)                # 跳过最后一行

skip_blank_lines：是否跳过空行，默认True，跳过空行

15）nrows：读取指定行，从文件第1行开始，经常用于较大的数据

16）parse_dates：指定日期时间解析器，默认为dateutil.parser.parser
date_parser = lambda x: pd.to_datetime(x, utc=True, format='%d-%b-%Y')
date_parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y')

parse_dates：指定日期时间列进行解析
pd.read_csv(data, index_col=0, parse_dates=True)    # parse_dates=True自动判断解析
pd.read_csv(data, parse_dates=['col'], date_parser=date_parser)
pd.read_csv(data, parse_dates={'dt': [1,4]})   # 将第1、4列合并解析成名为'dt'的时间类型列

17）encoding：字符编码格式，默认None

18）compression：压缩格式，对磁盘数据进行即时解压缩，默认infer，其他gzip、zip、bz2、xz和None(不进行解压缩)
pd.read_csv('csv_text.tar.gz', compression='gzip')

19）na_values：空值替换，该参数的值主要用于替换NA/NaN
pd.read_csv(data, index_col=None, na_values=['NA'])
"""

# 1.2）写入 to_csv()
# 常用参数解析：
'''
1）path_or_buf：文件路径
2）sep：输出文件的字段分隔符，默认','
3）na_rep：缺失值的字符串表示形式，默认为''
4）float_format：浮点数的格式化字符串
5）header：是否写出列名，默认True
6）columns：要写入的列，默认None
7）index：是否写入行索引名，默认True
8）encoding：编码格式
9）date_format：日期时间对象的格式字符串
10）line_terminator：表示行尾的字符，默认为os.linesep（当前平台的行终止符）
'''

# 2）excel文件：使用\t分割值
'''
读取常用参数：
Sheet名：默认第一个(索引0)；header：第几行为表头，默认0第一行；usecols：读取指定的列，如读取第一列和第二列[0,1]
df_excel = pd.read_excel(io="文件路径", sheet_name='Sheet名', header=0, usecols=[0,1])
写出常用参数：
Sheet名：默认第一个(索引0)；header：第几行为表头，默认0第一行；columns：指定写出的列；index：默认True显示行索引，一般设置为False
df_excel.to_excel(excel_writer='目标路径', sheet_name='Sheet名', header=0, columns=['列名', '...'], index=False)

with的用法：
with pd.ExcelWriter(f'...path/文件名.xlsx') as writer:
    df.to_excel(writer, sheet_name='sheet名', index=False, engine='xlsxwriter')
'''

# Excel 示例

# 2.1）读取 read_excel()、ExcelFile类
# ExcelFile类用于更方便地读取同一个文件的多张Sheet表格
with pd.ExcelFile('excel.xlsx') as xlsx:
    # sheet_names：返回Excel所有Sheet名列表
    print(xlsx.sheet_names)
    # parse(sheet, header, names, index_col, usecols)：支持参数基本与read_csv相同
    df1 = xlsx.parse('Sheet1')
    df2 = xlsx.parse('Sheet2')
    # 也可使用read_excel()
    # df1 = pd.read_excel(xlsx, 'Sheet1')
    # df2 = pd.read_excel(xlsx, 'Sheet2')

# 使用Sheet索引读取
pd.read_excel('excel.xlsx', 0, index_col=None, na_values=['NA'])
# 使用列表读取多张表格
pd.read_excel('path_to_file.xls', sheet_name=['Sheet1', 2])

# read_excel()支持读取常用参数与read_csv相同，例如
pd.read_excel('data', 'Sheet1', usecols='A,C:E')     # 指定列读取

# 2.2）写入 to_excel()、ExcelWriter类
# ExcelWriter作为Pandas库中一个用于将DataFrame数据写入Excel表格的工具，其使用十分灵活，极大地简化了数据处理和导出的流程
with pd.ExcelWriter('excel.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False, engine='xlsxwriter')
    df2.to_excel(writer, sheet_name='Sheet2', index=False, engine='xlsxwriter')

# engine：写入引擎，Pandas默认使用xlsxwriter

# 3）Json文件
'''
# 读取Json文件
# df5 = pd.read_json("sites.json")
# 使用Python JSON模块载入数据（见）
'''

# 4）读写压缩文件
'''
read_pickle()/to_pickle() + compression：读取或写入压缩序列化文件
compression：gzip、bz2、xz、zip、infer(默认使用推断)
'''
# df.to_pickle(file.gz, compression="gzip")
# pd.read_pickle(data, compression="gzip")

# 5）SQL查询
# SQLAlchemy模块提供了查询包装器的集合，以方便数据检索并减少对特定数据库API的依赖。数据库抽象由SQLAlchemy提供
# 5.1）读取：
'''
read_sql_table(table_name, cn, engine)
read_sql_query(sql, cn, engine)
read_sql(sql, cn, engine)
'''
# 5.2）写入：
'''
df.to_sql(table_name, cn, engine, index=False)
'''
from sqlalchemy import create_engine

# 从数据库URI创建引擎对象连接SQLAlchemy
engine = create_engine('url')
# 管理连接
with engine.connect() as conn, conn.begin():
    data = pd.read_sql_table('data', conn)

# SQLAlchemy详细介绍见：Python操作数据库/py_mysql.py

# Pandas数据加载、清洗与整理
'''
Pandas提供了多种方法来加载和处理数据，以便进行后续的分析和处理
以下是Pandas数据加载、清洗与整理的一些常用操作：
数据加载：使用pd.read_csv()函数可以从CSV文件中加载数据。你可以指定文件路径、分隔符、编码等参数来适应不同的数据源
缺失值处理：使用df.dropna()方法可以删除包含缺失值的行或列。使用df.fillna()方法可以填充缺失值
重复值处理：使用df.duplicated()方法可以判断是否存在重复值。使用df.drop_duplicates()方法可以删除重复值
数据转换：使用df.rename()方法可以重命名列名或索引名。使用df.replace()方法可以替换特定的值
数据合并：使用df.concat()函数可以合并多个DataFrame对象。使用df.merge()方法可以根据指定的列将两个DataFrame对象进行合并
'''

'''*--------------*----------------*----------------*--------------*--------------*'''
'''
# ※ DataFrame对象的操作 ※
df = pd.read_csv(r"C:\...\Desktop\property-data.csv")

# 1、查看数据

# 1）head(n)/tail(n) 读取前/后n行数据（默认前/后5行）
print(df.head(10))
print(df.tail(10))
# 2）values 以二维数组形式显示
print(df.values)
# 3）to_string() 以表格形式显示（为空的显示为NaN）
print(df.to_string())
# 4）info() 返回表格基本信息；dtypes 显示数据的信息，包括列名、数据类型、缺失值等
print(df3.info())
print(df.dtypes)

# 5）index 查看索引
print(df.index)         # Index([0, 1, 2], dtype='int64')
# 6）columns 查看列名
print(df.columns)       # Index(['A', 'B', 'C'], dtype='object')
# 7）describe() 查看数据的统计摘要
print(df.describe())
# 8）shape 查看轴维度
print(df.shape)        # (3, 3)

# 2、数据清洗及CRUD

# 1）数据类型转换
# Hive表 与 Python类型 对应关系：
# string    string
# double    float64
# bigint    int64
# A、改
df['col'] = df['col'].astype('类型')   # 指定列转换为指定类型
df = df.astype('类型')   # 将所有数据转换为指定类型

# 2）fillna(value) 将缺失值替换为指定的值，inplace=True用于修改源数据
# 对数据求和运算时，NaN（缺失值）将被视为零
df.fillna(0, inplace=True)   # df全部替换
df['col'].fillna(0, inplace=True)   # 指定列替换
# 3）replace('old_value', 'new_value') 将指定值替换为新值，regex=True用于指定使用正则
df['col'].replace('--', 0, inplace=True)
df['col'].replace('^[0-9]', '', inplace=True)

# 将A列的0和B列的0替换为空（df见16）中）
df.replace({'A': 0, 'B': 0}, np.nan)
# 替换多个不同值：将0和None都替换为空
df.replace([0, None], [np.nan, np.nan])

# B、删
# 4）dropna() 删除包含缺失值的行或列；参数：axis: 默认为0，表示行；1表示列 how='all': 只删除全为缺失值的行
# 5）df.duplicated() 检查是否有重复的数据
print(df.duplicated())
# 6）drop_duplicates() 去重
# 指定列去重，返回去重后的列序列
dup_col = df['col'].drop_duplicates()
# 按照指定列删除重复的数据；subset=['col']表示按照指定列去重；keep='first'表示取重复值的第一个
# df.drop_duplicates(subset=['col'], keep='first', inplace=True)
# 7）删除指定列、行：drop(labels=None,axis=0,index=None,columns=None,inplace=False)
# labels列名称或行索引 axis:0行1列 index指定行索引 columns指定列名
# df.drop(columns='col', axis=1, inplace=True)   删除指定列
# df.drop(index='索引值', inplace=True)   删除指定列
# 8）删除满足条件的行
# a、删除年份小于2000的所有行
df = df[df['year']>1999]
# b、只保留年份为2020和2021的所有行
df = df[(df['year'].isin([2020,2021]))]
# A、改
# 9）修改列字段类型
# df['col'] = df['col'].astype('指定type')
# df[['col1,...']] = df[['col1,...']].astype('指定type')   # 多列同时修改
# 10）重命名字段名
# df.rename(columns=rename_list, inplace=True)    # 方式1
# df.columns = rename_list                        # 方式2
# 重命名指定字段
# df.rename(columns={'old_col': 'new_col', ...})
# 11）将df第一行作为列名
df.columns = df.values.tolist()[0]    # 或：df.columns = np.array(df).tolist()[0]
df.drop(index=[0],inplace=True)       # 删除多余的第一行
# C、增
# 12）指定字段对该字段值计数，计数结果生成的列字段名默认为count，可重命名
# new_df = df['col'].value_counts().reset_index().rename(columns={'count': 'col_count'})
#      col  count                      col  col_count
# 0    v1   v1_count      ->      0    v1   v1_count
# 1    v2   v2_count              1    v2   v2_count
# ...  ...  ...                   ...  ...  ...
# 13）date_range(start,end,periods) 生成日期序列，常用参数：periods用于指定数据长度；start,end为开始结束日期
dates = pd.date_range("20230101", periods=6)    # 日期格式：yyyyMMdd、yyyy-MM-dd、d/M/yyyy
print(dates)
dates = pd.date_range("20230101", "20230106")
print(dates)      # 返回格式：DatetimeIndex
# 使用
for date in dates:
    print(date)   # 返回格式：yyyy-MM-dd hh:mm:ss
# 14）to_datetime() 将字符串、整数等格式的数据转换为时间戳
date_strs = ['2022-05-01', '2022-05-02', '2022-05-03']
dates = pd.to_datetime(date_strs)
print(dates)

# Timestamp()：yyyyMMdd转换为yyyy-MM-dd
print(pd.Timestamp('20221001'))    # 2022-10-01 00:00:00

# 15）新增一列，值为生成的list内容
df['新增列'] = pd.Series(list)

# assign()添加新列
print(df.assign(C=df['A'] + df['B']))

# A、改
# 16）Pandas处理文本字符串
df = pd.DataFrame({'A': [' abc', 'Ab ', 'BC', np.nan], 'B': [20, None, 10.0, np.nan]})
# 支持更多字符串处理函数
print(df['A'].str.upper())
print(df['A'].str.strip())

# D、查
# 17）检查缺失值，直观的返回True或False isna()
print(df.isna())

# pd.isnull()：检测对象的每个元素是否为空，直观的返回True或False
print(pd.isnull(np.nan))         # True
ls = [np.nan, 1, 0, None, '', []]
print(pd.isnull(ls).tolist())    # [True, False, False, True, False, False]

# A、改
# 18）Pandas提供的int64类型转换对NaN的支持：pd.Int64Dtype()
df['B_pd'] = df['B'].astype(pd.Int64Dtype())
df['B_Int'] = df['B'].astype('Int64')
print(df.to_string())
      A     B  B_pd  B_Int
0   abc  20.0    20     20
1   Ab    NaN  <NA>   <NA>
2    BC  10.0    10     10
3   NaN   NaN  <NA>   <NA>


# 3、数据选择和切片

# 1）数据选择：df['col']与df.col等效
# 选择指定列 df['column_name']
print(df['col'])
# 通过原有df（选择某些列）创建新的df
# new_df = df[['col1', 'col2', ...]]

# 行切片 df[sta,end]  [sta,end)
df = pd.DataFrame({
    'A': pd.Timestamp('20230502'),
    'B': pd.Series([1, 2, 3], index=list(range(3)), dtype='float64'),
    'C': np.array([4] * 3, dtype='int64')
})
print(df.to_string())
           A    B  C
0 2023-05-02  1.0  4
1 2023-05-02  2.0  4
2 2023-05-02  3.0  4
print(df[1: 2])
           A    B  C
1 2023-05-02  2.0  4

# 2）loc[]数据选择(按标签选择)  范围：[左闭右闭]
# 通过标签选择数据 df.loc[row_index, column_name]
print(df.loc[3, 'col'])
# 选择指定列值满足条件的数据
print(df.loc[df['col'] >= 200])
# 选择指定列值满足条件的数据，并指定列输出
print(df.loc[df['col'] >= 210, ['PID', 'NUM']])
# 获取指定行数据
print(df.loc[1])
# 选择所有行，选取指定列
print(df.loc[:, ['B', 'C']])
# 选择所有列，选取指定行（降维）
print(df.loc[1, :])
# 获取指定行和列对应的数据
print(df.loc[0][1])
# 快速访问标量：获取指定行列对应的数据
print(df.loc[1, 'B'])       # 2.0
print(df.at[1, 'B'])        # 2.0
# 3）iloc[]数据选择(按位置选择)  范围：[左闭右开)
# 切片 通过位置/索引选择数据（选择表部分数据） iloc[row_index(切片), column_name(切片)]
print(df.iloc[3, 1])
print(df.iloc[:, 2:5])

# 选取指定行列的数据
print(df.iloc[[0, 2], [1, 2]])
# 快速访问标量：指定行列对应的数据
print(df.iloc[1, 1])       # 2.0
print(df.iat[1, 1])        # 2.0

# 4）filter()数据过滤
# 选择指定的列 df.filter(items=['column_name1', 'column_name2'])
# print(df.filter(items=['col']))
# 选择列名匹配正则表达式的列
# print(df.filter(regex='^[0-9]'))
# 5）df[df['column_name'] > value]/df.query('column_name > value') 选择列中满足条件的数据
# print(df[df['col'] >= 20])
# print(df.query('col >= 20'))
# 筛选满足条件的索引
print(df[df['col'] >= 20].index.tolist())
# 6）布尔索引 isin()
print(df[df['B'].isin([2.0, 3.0])])

# 7）布尔索引 between()
print(df[df['B'].between(2, 3)])
           A    B  C
1 2023-05-02  2.0  4
2 2023-05-02  3.0  4


# 4、数据排序

# 1）sort_values('column_name') 按照指定列的值排序
print(df.sort_values(by='col', ascending=False))
# 2）rank() 计算新排序指标(新列)添加到df中
# df['new_col'] = df['col'].rank(ascending=False, method='dense')

# 5、数据分组与聚合

# 1）groupby('column_name') 按照指定列进行分组，多个列使用groupby(['col1',col2,...])
grouped = df.groupby(by='type_col')
# 2）取分组内指定类别的数据df
# group_df = grouped.get_group("col")
# 3）使用聚合函数统计某组元素数量
print(grouped.count()['col'])
# 4）transform变换函数: 计算新指标(新列)添加到df中
# new_df['新指标'] = new_df.groupby(by='col_m')['col_n'].transform('max')
# 5）分组数据遍历
for data, df in grouped:
    print(data)
    print(df.to_string())

# 6、数据合并

# 1）将多个数据df按照行或列进行合并 df = pd.concat([df1, df2])
# pd.concat(objs,axis=0,join='outer'): df拼接
# objs：需要连接的对象，如[df1,df2]，需要使用[]包裹
# axis=0（默认）：上下拼接，没有的使用NaN填充；axis=1：左右拼接，没有的使用NaN填充
# join='outer'（默认）：外连接，会保留两个表的全部信息；join='inner'：内连接：只保留两个表的公共信息

# 2）按照指定列将两个数据df进行合并
# pd.merge(left,right,how='inner',on=None,left_on=None,right_on=None): df连接
# left：左表  right：右表
# how='inner'(默认)：连接方式：inner内连接：两表公共；outer外连接：左连接与右连接的并集；left左连接：左表全部，右表匹配；right右连接：右表全部，左表匹配
# on=None(默认)：连接条件(连接列名)，必须同时存在于两个表中；若未指定，则以left和right列名的交集作为连接条件
# left_on,right_on：当两边字段名不同时，可以使用left_on,right_on设置连接

# 7、数据统计与描述
# 1）计算指定列非缺失值的数量
print(df['col'].count())
# 2）计算指定列的平均值
print(df['col'].mean())
# 3）计算指定列的中位数
print(df['col'].median())
# 4）计算指定列的众数
print(df['col'].mode())
# 5）计算基本统计信息，如均值、标准差、最小值、最大值等
print(df['col'].describe())

# skipna参数：True排除统计结果中的缺失数据，默认False
df_sum = df.sum(axis=1, skipna=True)
print(df_sum.to_string())

# 8、其它

# 1）转置

# 将df中某列值为str_list的列转置：
# print(np.array(df['col'].tolist()).T)
#         col                 0    1   ...
#         ['a','b']    =>     a    c   ...
#         ['c','d']           b    d   ...
#         ...

# 2）apply()与lambda表达式

# apply(func, axis) 将自定义函数应用于数据，对一行或一列进行操作（axis=0遍历列，axis=1遍历行）
df1 = pd.DataFrame([[3, 5], [4, 8]], columns=['A', 'B'])
print(df1.to_string())
# 计算各行sum
print(df1.apply(np.sum, axis=1))
# 计算各列sum
print(df1.apply(np.sum, axis=0))

# 自定义函数，实现添加新列为其他两列和
def add_col(x):
    return x.A + x.B

df1['C'] = df1.apply(lambda x: add_col(x), axis=1)
print(df1)

# 给列A的值全+1
df1['A'] = df1['A'].apply(lambda x: x + 1)
print(df1)

# 判断列A的值是否是偶数，用Y和N标注
df1['A'] = df1['A'].apply(lambda x: f"{x}/Y" if x % 2 == 0 else f"{x}/N")
print(df1)

# 3）赋值
# 使用Series赋值新增列数据，并使用索引对齐数据
s = pd.Series(['M', 'F', 'F'], index=[1, 0, 2])
df['G'] = s
print(df.to_string())
           A    B  C  G
0 2023-05-02  1.0  4  F
1 2023-05-02  2.0  4  M
2 2023-05-02  3.0  4  F
# 按标签赋值
df.loc[1, 'C'] = 5
df.at[1, 'C'] = 5
# 按位置赋值
df.iloc[2, 2] = 6
df.iat[2, 2] = 6
print(df.to_string())
           A    B  C  G
0 2023-05-02  1.0  4  F
1 2023-05-02  2.0  5  M
2 2023-05-02  3.0  6  F

4）时区序列
'''
ts = pd.Series(np.random.randn(3), pd.date_range('5/1/2023', periods=3, freq='D'))
print(ts)
'''
2023-05-01   -0.464566
2023-05-02    0.566463
2023-05-03   -0.881634
Freq: D, dtype: float64
'''
ts_utc = ts.tz_localize('UTC')
print(ts_utc)
'''
2023-05-01 00:00:00+00:00   -0.464566
2023-05-02 00:00:00+00:00    0.566463
2023-05-03 00:00:00+00:00   -0.881634
Freq: D, dtype: float64
'''
ts_us = ts_utc.tz_convert('US/Eastern')
print(ts_us)
'''
2023-04-30 20:00:00-04:00    0.308587
2023-05-01 20:00:00-04:00   -1.477437
2023-05-02 20:00:00-04:00    0.861031
Freq: D, dtype: float64
'''
ts_cn = ts_utc.tz_convert('Asia/Shanghai')
print(ts_cn)
'''
2023-05-01 08:00:00+08:00   -0.238885
2023-05-02 08:00:00+08:00   -0.513929
2023-05-03 08:00:00+08:00    0.533571
Freq: D, dtype: float64
'''
# 5）类别类型（Category）
# Pandas的DataFrame中可以包含类别类型的数据
df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "grade": ['A', 'A', 'B', 'A', 'B']})
print(df.to_string())
'''
   id grade
0   1     A
1   2     A
2   3     B
3   4     A
4   5     B
'''
# 用不同含义的名字重命名不同类型：Series.cat.categories
df['grade_ctg'] = df['grade'].copy().astype('category')
print(df['grade_ctg'])
'''
0    A
1    A
2    B
3    A
4    B
Name: grade_ctg, dtype: category
Categories (2, object): ['A', 'B']
'''
# # 此处报错：AttributeError: can't set attribute 'categories'
# df['grade_ctg'].cat.categories = ['very good', 'good']
# # 重新排序各类别
# df["grade_ctg"] = df["grade_ctg"].cat.set_categories(['good', 'very good'])
# print(df.to_string())

# 6）大型数据集处理加速操作
# 借助numexpr与bottleneck库的支持，Pandas可以加速特定类型的二进制数值与布尔操作
# 处理大型数据集时，这两个支持库加速效果非常明显；这两个支持库默认为启用状态，可用以下选项设置：
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

# 7）二进制操作：多维（DataFrame）与低维（Series）对象之间的广播/匹配机制
# DataFrame支持add()、sub()、mul()、div()及radd()、rsub()等方法执行二进制运算操作；fill_value属性可替换运算后的缺失值
df = pd.DataFrame({'one': [4, 10, np.nan], 'two': [3, 5, 2], 'three': [np.nan, 7, 6]})
print(df.to_string())
'''
    one  two  three
0   4.0    3    NaN
1  10.0    5    7.0
2   NaN    2    6.0
'''
df_sub = df.sub(df.iloc[1], axis='columns')
print(df_sub.to_string())
'''
   one  two  three
0 -6.0 -2.0    NaN
1  0.0  0.0    0.0
2  NaN -3.0   -1.0
'''
# 此处fill_value报错：fill_value not supported
df_sub = df.sub(df['two'], axis=0)
print(df_sub.to_string())
'''
   one  two  three
0  1.0    0    NaN
1  5.0    0    2.0
2  NaN    0    4.0
'''
# 若两个DataFrame同一个位置都有缺失值，其相加的和仍为NaN
df_add = df.add(df, fill_value=0)
print(df_add.to_string())
'''
    one  two  three
0   8.0    6    NaN
1  20.0   10   14.0
2   NaN    4   12.0
'''

# 8）Series与DataFrame支持比较操作
# eq等于、ne不等于、lt小于、gt大于、le小于等于、ge大于等于
# 布尔简化：empty、any()、all()、bool()
# empty df是否为空
print(df.empty)

# 9）合并重叠数据集
# 合并两个DataFrame对象，其中一个DataFrame中的缺失值将按指定条件用另一个DataFrame中类似标签中的数据进行填充
df1 = pd.DataFrame({'A': [1., np.nan, 3., 5., np.nan], 'B': [np.nan, 2., 3., np.nan, 6.]})
df2 = pd.DataFrame({'A': [5., 2., 4., np.nan, 3., 7.], 'B': [np.nan, np.nan, 3., 4., 6., 8.]})
print(df1.to_string())
print(df2.to_string())
'''
     A    B                A    B
0  1.0  NaN           0  5.0  NaN
1  NaN  2.0           1  2.0  NaN
2  3.0  3.0           2  4.0  3.0
3  5.0  NaN           3  NaN  4.0
4  NaN  6.0           4  3.0  6.0
                      5  7.0  8.0
'''
# combine_first()方法调用了更普适的DataFrame.combine()方法
df_com = df1.combine_first(df2)
print(df_com.to_string())
'''
     A    B
0  1.0  NaN
1  2.0  2.0
2  3.0  3.0
3  5.0  4.0
4  3.0  6.0
5  7.0  8.0
'''

# 10）pd.to_numeric(arg)
# pd.to_numeric(arg)：将参数转化为数字类型，默认返回dtype为int64或float64，具体取决于提供的数据
df = pd.DataFrame([{'a': '10', 'b': '66.6'}])
n1 = pd.to_numeric(df['a'])
n2 = pd.to_numeric(df['b'])
print(n1)   # dtype: int64
print(n2)   # dtype: float64

# 11）factorize标签编码
labels, uniques = df['ctg'].factorize()
print(list(labels))     # 标签编码，从0开始
print(uniques)          # 所有标签Index类型

'''*--------------*----------------*----------------*--------------*--------------*'''


# Pandas数据分析技巧
'''
Pandas提供了许多数据分析技巧和工具，以便进行数据处理、分析和可视化
以下是Pandas数据分析技巧的一些常见应用：
数据透视表：使用df.pivot_table()方法可以根据指定的行和列对数据进行透视，以便进行聚合和分析
时间序列分析：Pandas提供了一些用于处理时间序列数据的工具，如日期范围生成、日期索引设置、时间窗口划分等
数据可视化：Pandas结合了Matplotlib库，可以通过调用df.plot()方法进行数据可视化。你可以绘制折线图、柱状图、散点图等
'''






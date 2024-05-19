
import numpy as np
import pandas as pd

# Pandas操作常用文件API

# 1、CSV/文本文件

# read_table()
df = pd.read_table('path')
# 参数详解见官方文档：https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html
# 部分参数详解见下文read_csv()

# 1）读取 read_csv()
# read_table()、read_csv()常用参数解析：
"""
1）filepath_or_buffer：文件路径、url或具有方法的任何对象（如StringIO()）

2）sep：分隔符，默认','，可以使用正则表达式

3）header：表头，整型或整型列表类型，默认将第一行作为列名，数组可设置多级列；header会忽略空行和注释行，因此header=0表示第一行数据而非文件的第一行
对于没有表头的CSV，使用header=None，Pandas会为其添加默认表头(从0开始)

4）names：列名(指定表头)，列表类型，与数据一一对应，若文件不包含列名，则应该设置header=None；若文件包含列名，重新命名列名后可以删除原来列所在的行；列名列表中不允许有重复值

5）index_col：索引，DataFrame的行索引列表，可以是列编号或列名，Pandas不再使用首列作为索引，会自动使用以0开始的自然索引
pd.read_csv(data, index_col=n)                 # 指定第几列作为索引
pd.read_csv(data, index_col='col')             # 指定列名列作为索引
pd.read_csv(data, index_col=[n1, ...])         # 指定哪些列作为多级索引
pd.read_csv(data, index_col=['col1', ...])     # 指定哪些列作为多级索引

6）usecols：读取部分列，list类型或callable类型
pd.read_csv(data, usecols=[0,2,1])             # 按列索引读取指定列，与顺序无关
pd.read_csv(data, usecols=['col1', 'col3'])    # 按列名读取指定列，列名必须存在
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
from io import StringIO
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

19）na_values：将哪些值替换为空值，该参数的值主要用于替换NA/NaN
pd.read_csv(data, index_col=None, na_values=['NA'])

20）true_values、false_values：指定被视为True、False的值
"""
# 使用read_table()函数同样可以读取CSV文件，只需要指定分隔符,即可

# 2）写入 to_csv()
# 常用参数解析：
'''
1）path_or_buf：文件路径
2）sep：输出文件的字段分隔符，默认','
3）na_rep：将缺失值的替换为指定值，默认为''，如设置na_rep='0'
4）float_format：浮点数的格式化字符串
5）header：是否写出列名，默认True
6）columns：要写入的列，默认None
7）index：是否写入行索引名，默认True
8）encoding：编码格式
9）date_format：日期时间对象的格式字符串
10）line_terminator：表示行尾的字符，默认为os.linesep（当前平台的行终止符）
'''

# 2、Excel

# 1）读取 read_excel() ExcelFile类

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

# 设置多级行索引名称
# df.index.set_names(['lvl1', 'lvl2'])    # 例如2级行索引

# read_excel()支持参数与read_csv相同，例如
pd.read_excel('data', 'Sheet1', usecols='A,C:E')     # 指定列读取

# 2）写入 to_excel() ExcelWriter类
# ExcelWriter作为Pandas库中一个用于将DataFrame数据写入Excel表格的工具，其使用十分灵活，极大地简化了数据处理和导出的流程
with pd.ExcelWriter('excel.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# 特别参数：
# engine：写入引擎，Pandas默认使用xlsxwriter

# 3、读写压缩文件：read_pickle()/to_pickle() + compression：读取或写入压缩序列化文件

# compression：gzip、bz2、xz、zip、infer(默认使用推断)
# pd.read_pickle(path/buffer, compression="infer")
# df.to_pickle(file.pkl.zip, compression="infer")

# 报错：ValueError: If using all scalar values, you must pass an index
# 解决：添加：index=[0]
df = pd.DataFrame({'a': 1, 'b': 2}, index=[0])
df.to_pickle(r'C:\Users\cc\Desktop\file.pkl.zip', compression='infer')

# 4、SQL查询

# 1）读取：
'''
# 将SQL数据库表读入数据帧
sql.read_sql_table(table_name, con, schema, index_col, coerce_float, parse_dates, columns, chunksize)
# 将SQL查询读入数据帧
sql.read_sql_query(sql, con, index_col, coerce_float, params, parse_dates, chunksize, dtype)
# 将SQL查询或数据库表读入数据帧，read_sql()是read_sql_table()、read_sql_query()的统一方式
sql.read_sql(sql, con, index_col, coerce_float, params, parse_dates, columns, chunksize, dtype)
'''
# 2）写入：
'''
# 将数据帧记录写入数据库
sql.to_sql(df, name, con, schema, if_exists, index, index_label, chunksize, dtype, method, engine)
df.to_sql(df, name, con, schema, if_exists, index, index_label, chunksize, dtype, method, engine)
'''
# 读写参数详解：
'''
- table_name：数据库表名
- con：数据库连接的engine
- schema：数据库的引擎，若不设置则使用数据库的默认引擎，如MySQL的InnoDB
- index_col：选择某列作为index
- index_label：索引列的列标签
- coerce_float：将数字形式的字符串转换为float型，默认True
- parse_dates：将指定列日期型字符串转换为datetime型，与pd.to_datetime()函数功能类似，例如{'col': '%Y-%m-%d %H:%M:%S'}
- columns：选取的列
- chunksize：一次写入数据的行数，当数据量很大时，需要设置，否则会连接超时写入失败
- params：SQL传递的参数
- dtype：给列指定数据类型，字典类型
- name：数据库表名
- if_exists：fail(默认)：若表存在，不执行任何操作；replace：若表存在，先删除表再创建表，然后插入数据；append：若表存在，追加插入数据，不存在创建插入数据
- index：默认True，将df索引写入
- method：None(默认)：INSERT插入一个值（一行）；multi：INSERT插入多个值（多行）
- engine：数据库连接方式引擎，默认auto，也可指定sqlalchemy
'''

# SQLAlchemy模块提供了查询包装器的集合，以方便数据检索并减少对特定数据库API的依赖。数据库抽象由SQLAlchemy提供
# SQLAlchemy模块常用操作：见：Python操作数据库/py_mysql.py
from sqlalchemy import create_engine

# 从数据库URI创建引擎对象连接SQLAlchemy
engine = create_engine('url')
'''url取值
1）PostgreSQL：postgresql://user:pwd@localhost:port/db_name?charset=utf8
2）MySQL：mysql+pymysql://user:pwd@localhost:port/db_name?charset=utf8
3）Oracle：oracle://user:pwd@localhost:port/db_name?charset=utf8
4）Sqlite：sqlite:///db_name
'''
# 管理连接
with engine.connect() as conn, conn.begin():
    data = pd.read_sql_table('data', conn)

# 5、Parquet

# 1）读取 read_parquet()
'''
- path：文件路径或URL
- engine：引擎，默认auto，常见有fastparquet、pyarrow等
- columns：读取指定列
'''
df = pd.read_parquet('path')
# 2）写入 to_parquet()
'''
- path：文件路径或URL
- engine：引擎，默认auto
- compression：使用压缩格式，默认snappy
- index：文件是否包含数据帧的索引
'''
df.to_parquet('path', index=False)

# 6、Orc

# 1）读取
# pd.read_orc('path/url', columns)
# 2）写入
# df.to_orc('path', engine='pyarrow', index)

# 7、Json

# Json是最常用的标准数据格式之一，特别是Web数据的传输，通常在使用这些数据之前，需要对数据格式进行处理
'''
读取json文件：read_json()
写入json文件：to_json()
规范化(复杂格式)：json_normalize()
'''
# 1）写入 to_json()
df = pd.DataFrame(np.arange(12).reshape(3, 4), index=['red', 'green', 'blue'], columns=['up', 'right', 'down', 'left'])
print(df.to_string())
'''
       up  right  down  left
red     0      1     2     3
green   4      5     6     7
blue    8      9    10    11
'''
df.to_json('test.json')
'''
{"up":{"red":0,"green":4,"blue":8},"right":{"red":1,"green":5,"blue":9},"down":{"red":2,"green":6,"blue":10},"left":{"red":3,"green":7,"blue":11}}
'''
# 2）读取 read_json(json_file/url)
df1 = pd.read_json('test.json')
print(df1.to_string())
'''
       up  right  down  left
red     0      1     2     3
green   4      5     6     7
blue    8      9    10    11
'''
# 3）规范化：json_normalize()
# Json文件中的数据通常不是列表形式，因此，需要将字典结构的文件转成列表形式，这个过程就称为规范化
# Pandas库中的json_normalize()函数能够将字典或列表转换成表格，处理复杂结构的Json文件

# 案例1：Json对象列表
import json

# 读取json字符串
file = open("books.json", "r")
text = file.read()
text = json.loads(text)
print(text)

# json_normalize()函数会读取所有books作为键的元素的值，元素的所有属性将会转换为嵌套的列名称，而属性值将会转换为DataFrame元素，该函数使用一串递增的数字作为索引
'''
常用参数：
- data：json字符串
- record_path：解析Json对象中的嵌套列表，键列表或字符串类型
- meta：Json对象中的键，存在多层嵌套数据时可进行逐层解析
- sep：多层Key之间的分隔符，默认为'.'
'''
df2 = pd.json_normalize(text, 'books')
print(df2.to_string())
'''
    title  price
0     XML  23.56
1  Python  50.70
2   Numpy  12.30
3    Java  28.60
4   HTML5  31.35
5  Pandas  28.00
'''
# 将books位于同一级的其他键名的列表作为第三个参数传入可获得其他部分信息
df3 = pd.json_normalize(text, 'books', ['writer', 'nationality'])
print(df3.to_string())
'''
    title  price   writer nationality
0     XML  23.56     Ross         USA
1  Python  50.70     Ross         USA
2   Numpy  12.30     Ross         USA
3    Java  28.60  Bracket          UK
4   HTML5  31.35  Bracket          UK
5  Pandas  28.00  Bracket          UK
'''

# 案例2：嵌套列表的Json

# 读取json字符串
file = open("students.json", "r")
data = file.read()
data = json.loads(data)
print(data)

df1 = pd.json_normalize(data, record_path=['students'])
print(df1.to_string())
'''
     id   name  math  physics  chemistry
0  A001    Tom    60       66         61
1  A002  James    89       76         51
2  A003  Jenny    79       90         78
'''
df2 = pd.json_normalize(data, record_path=['students'], meta=['school_name', 'class'])
print(df2.to_string())
'''
     id   name  math  physics  chemistry         school_name   class
0  A001    Tom    60       66         61  ABC Primary School  Year 1
1  A002  James    89       76         51  ABC Primary School  Year 1
2  A003  Jenny    79       90         78  ABC Primary School  Year 1
'''
df3 = pd.json_normalize(
    data,
    record_path=['students'],
    meta=['class', ['info', 'president'], ['info', 'contacts', 'tel']]
)
print(df3.to_string())
'''
     id   name  math  physics  chemistry   class info.president info.contacts.tel
0  A001    Tom    60       66         61  Year 1           John         123456789
1  A002  James    89       76         51  Year 1           John         123456789
2  A003  Jenny    79       90         78  Year 1           John         123456789
'''



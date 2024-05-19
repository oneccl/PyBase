
# SQL On Pandas
# PandaSQL与DuckDB

# A、PandaSQL

import numpy as np
import pandas as pd

# 1、PandaSQL简介

# Pandas在数据处理方面提供了几乎全部的类SQL查询操作API，例如drop_duplicates()代表SQL中的union合并去重
# 但PandasAPI不如直接的SQL简洁易读，例如，Pandas还无法替代的操作之一是非等连接（查询连接条件包含非等号，如大于号、小于号等），这在SQL中非常简单，PandaSQL可以很好的解决这个问题
# PandaSQL是一个可以直接在Python中使用SQL语法查询Pandas数据框Dataframe的框架，PandaSQL底层调用PandasAPI
# 另外，Python虽然内置有SQLite数据库，但如果我们想使用SQL语句查询DataFrame就必须将原始数据先插入到SQLite
# 虽然PandaSQL允许我们在Pandas数据帧上运行SQL（SQLite语法）查询，但它的性能不如原生PandasAPI语法

# 安装：pip install -U pandasql

from pandasql import sqldf

# sqldf(query, env, db_uri)
# - query：使用DataFrame作为表的sql查询
# - env：环境globals()或locals()，允许sqldf访问Python环境中的全局或局部变量
# - db_uri：SQLAlchemy兼容的数据库URI，默认为sqlite:///:memory:
# 返回：返回查询结果DataFrame


def query(q: str, env=None):
    return sqldf(q, env=globals()) if env is None else sqldf(q, env=env)

# 2、PandasAPI与PandaSQL解决方案对比

# 1）数据准备
# 商品促销活动时期表
df_promotion = pd.DataFrame({
    "pdt_id": ["p01", "p02", "p03"],
    "start_dt": ["10-06-2023", "20-06-2023", "15-08-2023"],
    "end_dt": ["12-06-2023", "25-06-2023", "20-08-2023"]
})
# 商品交易数据表
df_trading = pd.DataFrame({
    "id": ["p01", "p01", "p02", "p02", "p02", "p03", "p03"],
    "trade_dt": ["11-06-2023", "20-06-2023", "15-08-2023", "22-06-2023", "11-06-2023", "17-08-2023", "29-08-2023"],
    "sales": [10, 20, 30, 22, 30, 20, 34]
})
# print(df_promotion.to_string())
# print(df_trading.to_string())

# 需求：查询促销期间商品的销售额

# 2）Pandas解决方案

# 合并
df_merge = pd.merge(df_promotion, df_trading, left_on="pdt_id", right_on="id")
# print(df_merge.to_string())
# 非等连接查询
df_query = df_merge[(df_merge["trade_dt"] >= df_merge["start_dt"]) & (df_merge["trade_dt"] <= df_merge["end_dt"])]
# 选择字段
df_res = df_query[["pdt_id", "start_dt", "end_dt", "trade_dt", "sales"]]
# print(df_res.to_string())
'''
  pdt_id    start_dt      end_dt    trade_dt  sales
0    p01  10-06-2023  12-06-2023  11-06-2023     10
1    p02  20-06-2023  25-06-2023  22-06-2023     22
2    p03  15-08-2023  20-08-2023  17-08-2023     20
'''

# 3）PandaSQL解决方案

sql = """
select pdt_id, start_dt, end_dt, trade_dt, sales from
df_promotion a join df_trading b
on a.pdt_id = b.id and b.trade_dt >= a.start_dt and b.trade_dt <= a.end_dt
"""

df = query(sql)
print(df.to_string())
'''
  pdt_id    start_dt      end_dt    trade_dt  sales
0    p01  10-06-2023  12-06-2023  11-06-2023     10
1    p02  20-06-2023  25-06-2023  22-06-2023     22
2    p03  15-08-2023  20-08-2023  17-08-2023     20
'''

# 3、PandaSQL支持的窗口函数
'''
聚合函数：sum()、count()、max()、min()、avg()
排序函数：rank()、dense_rank()、row_number()
平移函数：lead()、lag()、ntile()
取值函数：first_value()、last_value()
分布函数：percent_rank()、cume_dist()
'''

# 1）聚合函数
# 用于窗口分区内进行聚合
# partition by分区：数据全部显示（数量不变）
sql = """
select id, trade_dt,
count(*) over(partition by id) count,
sum(sales) over(partition by id) sum
from df_trading
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  count  sum
0  p01  11-06-2023      2   30
1  p01  20-06-2023      2   30
2  p02  15-08-2023      3   82
3  p02  22-06-2023      3   82
4  p02  11-06-2023      3   82
5  p03  17-08-2023      2   54
6  p03  29-08-2023      2   54
'''
# group by分组：select后面跟未分组的字段不会报错，该字段取原顺序的第一个值（数量减少）
sql = """
select id, trade_dt,
count(*) count,
sum(sales) sum
from df_trading
group by id
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  count  sum
0  p01  11-06-2023      2   30
1  p02  15-08-2023      3   82
2  p03  17-08-2023      2   54
'''

# 2）排序函数
# rank()：总数不变，值相同重复，如1 2 2 4 ...
# dense_rank()：总数减少，值相同重复，如1 2 2 3 ...
# row_number()：始终按顺序排名，如1 2 3 4 ...
sql = """select *, row_number() over(partition by id order by sales desc) rk from df_trading"""
df_rk = query(sql)
print(df_rk.to_string())
'''
    id    trade_dt  sales  rk
0  p01  20-06-2023     20   1
1  p01  11-06-2023     10   2
2  p02  15-08-2023     30   1
3  p02  11-06-2023     30   2
4  p02  22-06-2023     22   3
5  p03  29-08-2023     34   1
6  p03  17-08-2023     20   2
'''

# 3）平移函数
# lead(col,n,default)：窗口分区内从上向下平移n行数据
# lag(col,n,default)：窗口分区内从下向上平移n行数据
# ntile(n)：将有序分区平均划分n组，每组都有编号，编号从1开始，返回组的编号（n必须为int类型）
# rows between ... and ...：指定窗口范围
# - n preceding ：往前n行
# - n following：往后n行
# - current row：当前行
# - unbounded preceding：当前窗口首行
# - unbounded following：当前窗口末行
sql = """
select id, trade_dt, sales,
lead(sales, 1, null) over(partition by id order by trade_dt asc) next_sales,
lag(sales, 1, null) over(partition by id order by trade_dt asc) last_sales
from df_trading
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  sales  next_sales  last_sales
0  p01  11-06-2023     10        20.0         NaN
1  p01  20-06-2023     20         NaN        10.0
2  p02  11-06-2023     30        30.0         NaN
3  p02  15-08-2023     30        22.0        30.0
4  p02  22-06-2023     22         NaN        30.0
5  p03  17-08-2023     20        34.0         NaN
6  p03  29-08-2023     34         NaN        20.0
'''
# 将根据trade_dt排序后的数据平均分成2组，返回第一组数据
sql = """
select * from (
    select id, trade_dt, sales, ntile(2) over(order by trade_dt asc) sorted from df_trading
) where sorted = 1
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  sales  sorted
0  p01  11-06-2023     10       1
1  p02  11-06-2023     30       1
2  p02  15-08-2023     30       1
3  p03  17-08-2023     20       1
'''

# 4）取值函数
# first_value(col)：获取窗口分区内第一行
# last_value(col)：获取窗口分区内最后一行
# 结果总数不变，会重复
sql = """
select id, trade_dt, sales,
first_value(sales) over(partition by id order by sales desc) first_info
from df_trading
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  sales  first_info
0  p01  20-06-2023     20          20
1  p01  11-06-2023     10          20
2  p02  15-08-2023     30          30
3  p02  11-06-2023     30          30
4  p02  22-06-2023     22          30
5  p03  29-08-2023     34          34
6  p03  17-08-2023     20          34
'''

# 5）分布函数
# percent_rank()：按照排名计算百分比，即排名位于区间[0,1]，其中区间内第一名为值0，最后一名值为1
# cume_dist()：区间內大于等于当前排名的行数占区间内总数的比例
sql = """
select *,
percent_rank() over(partition by id order by sales desc) percent_rk,
cume_dist() over(partition by id order by sales desc) scale
from df_rk
"""
df = query(sql)
print(df.to_string())
'''
    id    trade_dt  sales  rk  percent_rk     scale
0  p01  20-06-2023     20   1         0.0  0.500000
1  p01  11-06-2023     10   2         1.0  1.000000
2  p02  15-08-2023     30   1         0.0  0.666667
3  p02  11-06-2023     30   2         0.0  0.666667
4  p02  22-06-2023     22   3         1.0  1.000000
5  p03  29-08-2023     34   1         0.0  0.500000
6  p03  17-08-2023     20   2         1.0  1.000000
'''


# 4、PandaSQL综合使用
# 占比、同比、环比、移动平均线

# 某商品的销售日期及数量如下
dfx = pd.DataFrame({
    "dt": pd.date_range("20230501", periods=12, freq="W"),
    "sales": [40, 80, 50, 10, 30, 45, 80, 90, 20, 60, 35, 70]
})
print(dfx.to_string())

# 1）占比：月销量占本年总销量比
sql = """
-- 除法运算：需要将除数或被除数转为浮点型，否则结果始终为0
select dt, month_sum, year_sum, round(month_sum*1.0 / year_sum, 4) ratio from (
    select
    -- SQLite不支持concat()函数，需要使用'||'连接
    substr(dt, 1, 7) || '-01' dt,
    -- 月销量
    sum(sales) over(partition by substr(dt, 1, 7)) month_sum,
    -- 年总销量
    sum(sales) over(partition by substr(dt, 1, 4)) year_sum,
    -- row_number()用于去重，也可以使用distinct
    row_number() over(partition by substr(dt, 1, 7)) rk
    from dfx
) t
where rk = 1
"""
df = query(sql)
print(df.to_string())
'''
           dt  month_sum  year_sum   ratio
0  2023-05-01        180       610  0.2951
1  2023-06-01        245       610  0.4016
2  2023-07-01        185       610  0.3033
'''

# 2）同比、环比变化和增长率（以同比为例）
# 方式1：直接使用lag()窗口函数
sql = """
select t2.dt, t2.month_sum, t2.last_month_sum,
t2.month_sum - t2.last_month_sum month_sum_change,
round((t2.month_sum - t2.last_month_sum)*1.0 / t2.last_month_sum, 2) increase_rate
from (
    select *,
    lag(t1.month_sum, 1) over(order by t1.dt asc) as last_month_sum
    from (
        select distinct
        substr(dt, 1, 7) || '-01' dt,
        sum(sales) over(partition by substr(dt, 1, 7)) month_sum
        from dfx
    ) t1
) t2
"""
df = query(sql)
print(df.to_string())
'''
           dt  month_sum  last_month_sum  month_sum_change  increase_rate
0  2023-05-01        180             NaN               NaN            NaN
1  2023-06-01        245           180.0              65.0           0.36
2  2023-07-01        185           245.0             -60.0          -0.24
'''

# 方式2：使用类似Hive的add_months()函数+join
sql = """
with t1 as (
    select distinct
    substr(dt, 1, 7) || '-01' dt,
    sum(sales) over(partition by substr(dt, 1, 7)) month_sum
    from dfx
)
select
t1.dt, t1.month_sum, t2.month_sum last_month_sum,
t1.month_sum - t2.month_sum month_sum_change,
round((t1.month_sum - t2.month_sum)*1.0 / t2.month_sum, 2) increase_rate
from t1
left join (
    -- SQLite日期加减函数：date(col, '±n day/month')、datetime(col, '±n day/month')
    select *, date(dt, '+1 month') tmp_date from t1
) t2
on date(t1.dt) = t2.tmp_date
"""
df = query(sql)
print(df.to_string())
'''
           dt  month_sum  last_month_sum  month_sum_change  increase_rate
0  2023-05-01        180             NaN               NaN            NaN
1  2023-06-01        245           180.0              65.0           0.36
2  2023-07-01        185           245.0             -60.0          -0.24
'''

# 3）计算近3周和近7周的总销量、平均销量、前后3周销量的移动平均线
sql = """
select date(dt) dt, sales,
sum(sales) over w1 as latest3,
round(avg(sales) over w1, 2) as average3,
sum(sales) over w2 as latest7,
round(avg(sales) over w2, 2) as average7,
round(avg(sales) over w3, 2) as perfol3
from dfx
window
    w1 as (order by dt asc rows between 2 preceding and current row),
    w2 as (order by dt asc rows between 6 preceding and current row),
    w3 as (order by dt asc rows between 1 preceding and 1 following)
"""
df = query(sql)
print(df.to_string())
'''
            dt  sales  latest3  average3  latest7  average7  perfol3
0   2023-05-07     40       40     40.00       40     40.00    60.00
1   2023-05-14     80      120     60.00      120     60.00    56.67
2   2023-05-21     50      170     56.67      170     56.67    46.67
3   2023-05-28     10      140     46.67      180     45.00    30.00
4   2023-06-04     30       90     30.00      210     42.00    28.33
5   2023-06-11     45       85     28.33      255     42.50    51.67
6   2023-06-18     80      155     51.67      335     47.86    71.67
7   2023-06-25     90      215     71.67      385     55.00    63.33
8   2023-07-02     20      190     63.33      325     46.43    56.67
9   2023-07-09     60      170     56.67      335     47.86    38.33
10  2023-07-16     35      115     38.33      360     51.43    55.00
11  2023-07-23     70      165     55.00      400     57.14    52.50
'''

# rows between与range between窗口的区别
# rows between V_sta and V_end
# 实际行：按实际行加减
# range between V_sta and V_end
# 逻辑行：对order by后的当前行字段值分别减去V_sta和加上V_end，得到一个区间[V_cur-V_sta, V_cur+V_end]，计算该区间（窗口）范围内的聚合值
sql = """
select date(dt) dt, sales,
sum(sales) over (order by sales asc range between 10 preceding and 10 following) as res
from dfx
"""
df = query(sql)
print(df.to_string())
'''说明：例如res=125：当前行排序字段sales=30，则窗口范围[20,40]，结果需要计算sales在该范围中的和：20+30+35+40=125
            dt  sales   res
0   2023-05-28     10    30
1   2023-07-02     20    60
2   2023-06-04     30   125
3   2023-07-16     35   150
4   2023-05-07     40   200
5   2023-06-11     45   170
6   2023-05-21     50   195
7   2023-07-09     60   180
8   2023-07-23     70   290
9   2023-05-14     80   320
10  2023-06-18     80   320
11  2023-06-25     90   250
'''

# 通过上面PandasAPI与PandaSQL的对比，我们发现Python虽好，但很明显，在对数据进行分析时，SQL显得更优雅和易于理解，SQL语法更通俗、直观


# B、DuckDB

# 1、DuckDB简介
# DuckDB是一个开源的内存中的OLAP数据库管理系统（DBMS），DuckDB旨在支持分析查询工作负载，也称为联机分析处理（OLAP）
# DuckDB某种意义上是SQLite/OLAP的等效工具，DuckDB支持结构化查询语言（SQL），并支持在Pandas DataFrame上执行SQL查询
# SQLite主要专注于事务性 (OLTP) 工作负载。它的执行引擎处理以B-Tree存储格式存储的行。而DuckDB对标SQLite支持分析（OLAP）工作负载。DuckDB能够填补OLAP领域嵌入式DBMS的空缺

# DuckDB官网：https://duckdb.org/

# DuckDB的主要特性：
'''
开源和免费，宽松的MIT许可证，可扩展性（可运行在Windows，Linux，macOS等）
安装简单，嵌入式操作（不作为单独的进程运行，而是完全嵌入在主机进程中），无DBMS服务器软件
无依赖关系（无论是编译还是运行时），单文件构建，方便部署、集成和维护
支持多语言，深度集成Python、R，为Java、C/C++等提供API，以实现高效的交互式数据分析
支持复杂的SQL查询，包括多个大型表间的联接，窗口函数等，并提供优化的多版本并发控制（MVCC）和事务保证（ACID）
支持大规模数据仓库，数据支持存储在持久性单文件数据库中，支持二级索引
OLAP查询性能高，采用最先进的技术矢量化（集成Apache Arrow：零拷贝、列式存储、谓词下推）和实时查询执行引擎（并发引擎），某些场景的性能优于Clickhouse
能够与Parquet、ORC等兼容，内置加载器，支持直接使用Parquet、CSV、JSON等文件查询
'''

# 安装：pip install duckdb==0.9.1
# 官方文档：https://duckdb.org/docs/guides

# 2、SQL操作（SQL On Pandas）
import duckdb

# sql = "select * from df_res"
# # duckdb.sql(sql,connect=None)或duckdb.query(sql,connect=None)，返回SQL查询对象
# duckdb.sql(sql).show()
# duckdb.query(sql).show()
# '''
# ┌─────────┬────────────┬────────────┬────────────┬───────┐
# │ pdt_id  │  start_dt  │   end_dt   │  trade_dt  │ sales │
# │ varchar │  varchar   │  varchar   │  varchar   │ int64 │
# ├─────────┼────────────┼────────────┼────────────┼───────┤
# │ p01     │ 10-06-2023 │ 12-06-2023 │ 11-06-2023 │    10 │
# │ p02     │ 20-06-2023 │ 25-06-2023 │ 22-06-2023 │    22 │
# │ p03     │ 15-08-2023 │ 20-08-2023 │ 17-08-2023 │    20 │
# └─────────┴────────────┴────────────┴────────────┴───────┘
# '''
# # 将SQL查询对象转换为Pandas DataFrame
# print(duckdb.query(sql).df().to_string())
# '''
#   pdt_id    start_dt      end_dt    trade_dt  sales
# 0    p01  10-06-2023  12-06-2023  11-06-2023     10
# 1    p02  20-06-2023  25-06-2023  22-06-2023     22
# 2    p03  15-08-2023  20-08-2023  17-08-2023     20
# '''
# # 将SQL查询对象转换为Python对象
# print(duckdb.sql(sql).fetchall())
# '''
# [('p01', '10-06-2023', '12-06-2023', '11-06-2023', 10), ('p02', '20-06-2023', '25-06-2023', '22-06-2023', 22), ('p03', '15-08-2023', '20-08-2023', '17-08-2023', 20)]
# '''
#
# # 默认情况下，DuckDB使用全局内存中连接。程序关闭后，任何数据都将丢失。可以使用connect()函数创建与持久数据库的连接，DuckDB会在桌面创建持久化数据文件
# con = duckdb.connect(r'C:\Users\cc\Desktop\file.db')
# con.sql("create table test as select * from df_res")
# con.sql("insert into test select * from df_res")
# con.sql("select * from test").show()
#
# # 参数化查询：con.execute(sql,params)
# result = con.execute("select * from test where pdt_id = ?", ('p01',)).fetchall()
# print(result)
#
#
# # 3、逻辑SQL（DSL on Pandas）
# # 连接到内存中的数据库
# con = duckdb.connect()
# df_in = pd.DataFrame({"A": ['onr', 'two', 'three'], "B": [3, 2, 5]})
# # 从数据框架创建一个DuckDB关系
# rel = con.from_df(df_in)
# # DSL
# df_out = rel.filter('B >= 3').project('A, B, B*2 as C').order('B desc').df()
# print(df_out.to_string())
# '''
#        A  B   C
# 0  three  5  10
# 1    onr  3   6
# '''

# 4、DuckDB on Apache Arrow
# DuckDB支持查询多种不同类型的Apache Arrow对象
# 官方文档：https://duckdb.org/docs/guides/python/sql_on_arrow

# 5、DuckDB On fsspec Filesystems
# DuckDB支持查询对fsspec支持的文件系统中的数据，此功能仅支持在Python客户端中使用
# fsspec是Python的文件系统接口（规范），fsspec详细介绍：https://filesystem-spec.readthedocs.io/en/latest/
# 如果对象存储（云存储）实现了fsspec文件系统接口（规范），那么fsspec允许你像处理本地数据一样处理对象存储中的数据
# 安装fsspec：pip install fsspec
# 例如，安装Google云存储fsspec的Python实现(gcsfs)：pip install gcsfs
# 注册需要查询的GCS文件系统：duckdb.register_filesystem(filesystem('gcs'))
# 然后就可以使用了

# 腾讯云COS对象存储演示：
# 安装腾讯云存储fsspec的Python实现(cosfs)：
# pip install cosfs --upgrade -i https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --extra-index-url https://mirrors.tencent.com/pypi/simple/
# 腾讯云cosfs项目GitHub开源地址：https://github.com/Panxing4game/cosfs
#
from fsspec import filesystem

# 注册要查询的COS文件系统
duckdb.register_filesystem(filesystem('cosn'))

sql = '''select * from read_csv_auto('cosn://bucket/../data.csv')'''
duckdb.sql(sql).show()

# # 6、文件数据导入导出
#
# # 1）CSV
#
# # CSV导入：read_csv_auto(path)、read_csv(path,delim,header=true,columns={'col': 'type', ...})
# # read_csv_auto()：自动检测（分隔符、转义、列类型、标题行等）
# # read_csv()：delim指定分隔符，header=true首行为列，columns字段类型
# sql = """select * from read_csv_auto('C:/Users/cc/Desktop/input.csv')"""
# duckdb.sql(sql).show()
#
# # 将文件数据插入到已有表中
# sql = """create table dft (id varchar, name varchar, age int64)"""
# duckdb.sql(sql)
# sql = """insert into dft select * from read_csv_auto('C:/Users/cc/Desktop/input.csv')"""
# duckdb.sql(sql)
# sql = """select * from dft"""
# dft = duckdb.sql(sql).df()
# print(dft.to_string())
#
# # CSV导出：copy table/select to path (header, delimiter '分隔符')
# sql = """copy (select * from df_out) to 'C:/Users/cc/Desktop/output.csv' (header, delimiter ',')"""
# duckdb.sql(sql)
#
#
# # 2）Parquet
#
# # Parquet导入：read_parquet()
# sql = """select * from read_parquet('C:/Users/cc/Desktop/input.parquet')"""
# duckdb.sql(sql).show()
#
# # Parquet导出：copy table/select to path (format parquet)
# sql = """copy (select * from df_out) to 'C:/Users/cc/Desktop/output.parquet' (format parquet)"""
# duckdb.sql(sql)
#
# # HTTP/HTTPS Parquet补充：
# # 若想通过HTTP（S）加载Parquet文件，需要使用SQL命令安装并加载httpfs扩展，这只需要运行一次
# # 安装：install httpfs;
# # 加载：load httpfs;
# # 使用：select * from read_parquet('https://../file.parquet');
#
# # 3）JSON
#
# # json导入：read_json_auto(path)
# sql = """select * from read_json_auto('C:/Users/cc/Desktop/input.json')"""
# duckdb.sql(sql).show()
#
# # json导出：copy table/select to path
# sql = """copy (select * from df_out) to 'C:/Users/cc/Desktop/output.json'"""
# duckdb.sql(sql)
#
# # 4）Excel
#
# # 若想要从Excel文件读取数据，则需要使用SQL命令安装并加载spatial扩展，这只需要运行一次
# # 安装：install spatial;
# # 加载：load spatial;
#
# duckdb.sql('install spatial')
# duckdb.sql('load spatial')
#
# # Excel导入：st_read(path, layer=sheet_name)
# sql = """select * from st_read('C:/Users/cc/Desktop/input.xlsx', layer='Sheet1')"""
# duckdb.sql(sql).show()
#
# # Excel导出：copy table/select to path with (format gdal, driver 'xlsx')
# sql = """copy (select * from df_out) to 'C:/Users/cc/Desktop/output.xlsx' with (format gdal, driver 'xlsx')"""
# duckdb.sql(sql)
#
# # 5）多文件读取
# # DuckDB可以使用glob语法或通过提供要读取的文件列表同时读取多个文件（CSV、Parquet、JSON），相当于union all
# # 注意：多个文件列结构必须相同该操作才有效，默认按照读取提供的第一个文件的列结构
# sql = """select * from read_csv_auto('path/../*.csv')"""
# # filename=true：向结果添加额外的列，用于指示当前行来自哪个文件，值为文件名
# # union_by_name=true：统一列结构，多个文件列结构不相同时使用，列缺失值显示为NULL
# sql = """select * from read_csv_auto(['path1', 'path2'], union_by_name=true, filename=true)"""
# duckdb.sql(sql).show()
#
# # 7、DuckDB扩展数据源
#
# # 1）PostgreSQL
#
# # 默认情况下，PostgreSQL扩展postgres将在首次使用时自动加载。如果您希望显式执行此操作，则可以使用以下命令完成此操作：
# # 安装：install postgres;
# # 加载：load postgres;
# # postgres允许DuckDB直接从正在运行的PostgreSQL实例读取数据。数据可以直接从底层的PostgreSQL表中查询，也可以读入DuckDB表
#
# # 使用：
# # 使PostgreSQL数据库可供DuckDB访问
# # postgres_attach(连接参数,source_schema,sink_schema)：从源模式加载到目标模式
# # - 连接参数：字符串类型，可以传递选择数据库名称
# # - source_schema：PostgreSQL中用于获取表的非标准模式名称，默认public
# # - sink_schema：使用DuckDB中的模式名来创建视图。默认main
# sql = """call postgres_attach('dbname=postgres user=root host=localhost', source_schema='public', sink_schema='abc')"""
# duckdb.sql(sql)
#
# # 列出数据库中的表在DuckDB中注册的视图：pragma show_tables;
# # 然后，可以使用SQL正常查询这些视图
#
# # 查询
# # postgres_scan('', schema, table)：使用默认连接从source_schema中查询table
# sql = """select * from postgres_scan('', 'public', 'table')"""
# duckdb.sql(sql).show()
#
# # 2）MySQL
#
# # mysql_scanner扩展允许DuckDB直接从正在运行的MySQL实例读取和写入数据。可以直接从底层MySQL数据库查询数据。数据可以从MySQL表加载到DuckDB表中
# # 官方文档：https://duckdb.org/docs/extensions/mysql_scanner
#
# # 8、DuckDB SQL语法：https://duckdb.org/docs/sql/introduction
#
# # 9、DuckDB客户端接口：https://duckdb.org/docs/api/overview
#
# # 10、DuckDB分区与谓词下推：https://duckdb.org/docs/data/partitioning/hive_partitioning


# Pandas、PandaSQL、DuckDB性能比较

# Pandas和DuckDB性能相差不大
# PandaSQL性能不如Pandas和DuckDB，Pandas和DuckDB性能大概是PandaSQL的7倍




# Python3连接MySQL主要有2种方式：mysql-connector模块和PyMySQL模块
"""
mysql-connector: Oracle官方支持、略慢、不兼容MySQLdb
PyMySQL: 比mysql-connector略快，兼容MySQLdb
"""
'''
注意：对于mysql-connector模块，若MySQL是8.x版本，密码插件验证方式发生了变化，需要做如下改变（否则报错）：
1）修改my.ini配置：
default_authentication_plugin=caching_sha2_password
2）修改密码：
alter user 'root'@'localhost' identified with caching_sha2_password by 'password';
'''

# 1、mysql-connector
# 安装：pip install mysql-connector
import mysql.connector

# 使用步骤：
# 1）创建连接
cn = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='py_mysql'
)
# 2）创建游标对象：游标对象可以执行SQL语句并获取执行结果
cursor = cn.cursor()
# 3）执行SQL语句：cursor.execute('sql', [args])
# 4）提交更改：执行更新操作（插入、更新或删除）后，需要使用连接对象的commit()方法来提交更改，若发生错误，可使用rollback()方法进行事务回滚
# 5）获取执行结果：执行查询操作后，可以使用游标对象的fetchall()方法获取查询结果
# 6）关闭游标和连接，释放资源：cursor.close()、cn.close()

# 1.1、创建数据库、创建表
cursor.execute("create table stu(name varchar(20), age int)")

# 1.2、CRUD
# 1）增
# a、插入一条
sql = "insert into stu (name, age) values (%s, %s)"
val = ('Tom', 18)
cursor.execute(sql, val)
cn.commit()
# b、批量插入：executemany()
vals = [('Bob', 18), ('Jerry', 18)]
cursor.executemany(sql, vals)
cn.commit()
print(cursor.rowcount)          # 返回影响行数

# 2）改
sql = "update stu set age=%s where name=%s"
val = (19, 'Bob')
cursor.execute(sql, val)
cn.commit()

# 3）删
sql = "delete from stu where name=%s"
val = ('Tom',)
cursor.execute(sql, val)
cn.commit()

# 4）查
sql = "select * from stu where name !='' order by age desc limit 3"
cursor.execute(sql)
print(cursor.fetchone())       # 获取下一个结果
result = cursor.fetchall()     # 获取所有结果
for row in result:
    print(row)                 # 以元组形式显示一行数据

# 释放资源
cursor.close()
cn.close()


# 2、PyMySQL
# 安装：pip install PyMySQL
import pymysql

# 1）创建连接
cn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='py_mysql'
)
# 2）创建游标对象：游标对象可以执行SQL语句并获取执行结果
cursor = cn.cursor()
# 3）执行SQL语句：cursor.execute('sql', [args])
# 4）提交更改：执行更新操作（插入、更新或删除）后，需要使用连接对象的commit()方法来提交更改，若发生错误，可使用rollback()方法进行事务回滚
# 5）获取执行结果：执行查询操作后，可以使用游标对象的fetchall()方法获取查询结果
# 6）关闭游标和连接，释放资源：cursor.close()、cn.close()

# 1.1、创建数据库、创建表
cursor.execute("create table stu(name varchar(20), age int)")

# 1.2、CRUD
# 1）增
# a、插入一条
sql = "insert into stu (name, age) values (%s, %s)"
val = ('Tom', 18)
cursor.execute(sql, val)
cn.commit()
# b、批量插入：executemany()
vals = [('Bob', 18), ('Jerry', 18)]
cursor.executemany(sql, vals)
cn.commit()
print(cursor.rowcount)          # 返回影响行数

# 2）改
sql = "update stu set age=%s where name=%s"
val = (19, 'Bob')
try:
    cursor.execute(sql, val)
    cn.commit()
except:
    cn.rollback()

# 3）删
sql = "delete from stu where name=%s"
val = ('Tom',)
cursor.execute(sql, val)
cn.commit()

# 4）查
sql = "select * from stu where name !='' order by age desc limit 3"
cursor.execute(sql)
print(cursor.fetchone())       # 获取下一个结果
result = cursor.fetchall()     # 获取所有结果
for row in result:
    print(row)                 # 以元组形式显示一行数据

# 释放资源
cursor.close()
cn.close()


# SQLAlchemy：ORM框架

# SQLAlchemy模块提供了查询包装器的集合，以方便数据检索并减少对特定数据库API的依赖。数据库抽象由SQLAlchemy提供
# SQLAlchemy是PythonSQL工具包和对象关系映射器，为应用程序开发人员提供SQL的全部功能和灵活性
# SQLAlchemy提供了一整套众所周知的企业级持久性模式，专为高效和高性能的数据库访问而设计
# 官网：https://www.sqlalchemy.org/
# 安装：pip install sqlalchemy

# 1、基本使用（CRUD）：
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 从数据库URI创建引擎对象连接：create_engine('url')
'''url取值
1）PostgreSQL：postgresql://user:pwd@localhost:port/db_name?charset=utf8
2）MySQL：mysql+pymysql://user:pwd@localhost:port/db_name?charset=utf8
3）Oracle：oracle://user:pwd@localhost:port/db_name?charset=utf8
4）Sqlite：sqlite:///db_name
'''

# 连接数据库
# echo：默认False，表示不打印执行SQL语句的详细的信息
engine = create_engine("sqlite:///db_name", echo=True)
Base = declarative_base()

# 定义数据表模型
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

# 创建数据表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

# 1）增
# 插入数据（需要commit确认才会写入数据库），返回影响行数
user1 = User(name="Alice", age=18)
user2 = User(name="Bob", age=20)
# 插入多条数据
session.add_all([user1, user2])
# 插入一条数据
# session.add(user1)
session.commit()

# 2）查
# 查询数据（不需要commit）
# 查询所有
users = session.query(User).all()
for user in users:
    print(user.name, user.age)
# 查询单个
user = session.query(User).filter_by(id='100').first()
print(user)

# 3）改
# 更新数据（需要commit确认才会更新数据库）
# 方式1：先查询，再更新
user = session.query(User).filter_by(id='100').first()
user.name = "Tom"
user.age = 18
session.commit()
# 方式2：返回影响行数
update = session.query(User).filter(User.id=='100').update({User.name: "Tom", User.age: 18})
session.commit()

# 4）删
# 删除数据（需要commit确认才会删除）
# 方式1：先查询，再删除
mark = session.query(User).filter_by(id='100').first()
session.delete(mark)
session.commit()
# 方式2：返回影响行数
session.query(User).filter(User.id=='100').delete()
session.commit()

# 注意：跨表的update/delete等函数中要注明参数synchronize_session=False

# 关闭会话
session.close()

# 2、SQLAlchemy模块常用操作：

# 将SQL查询结果转换为DataFrame对象：

import pandas as pd

df_res = pd.read_sql(sql, con=cn)
print(df_res.to_string())

# 将DataFrame对象写入MySQL：

from sqlalchemy import create_engine
from pandas.io import sql

# 确定运行环境上有pymysql
# sql_cn = create_engine(rf"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8")
# sql.to_sql(df, tab_name, sql_cn, schema=database, if_exists='append', index=False)

# 文件（如CSV）写入MySQL：

import numpy as np

# 确定运行环境上有pymysql
# sql_cn = create_engine(rf"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8")
# df_csv = pd.read_csv('file_name', sep='\t', dtype=np.str)
# sql.to_sql(df_csv, tab_name, sql_cn, schema=database, if_exists='append', index=False)

# SQLAlchemy模块的事务支持：

from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session

# 创建数据库连接引擎和Session
'''
engine = create_engine(rf"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8")
Session = sessionmaker(bind=engine)   # Session工厂
session = Session()   # Session会话，线程不安全，多个线程共享一个Session
# session = scoped_session(Session)   # 单例Session会话，每个线程一个Session，多个线程不共享Session

try:
    # 开始事务
    session.begin()
    # 执行数据库操作
    # ...
    # 提交事务
    session.commit()
except:
    # 发生异常时回滚
    session.rollback()
finally:
    # 关闭Session
    session.close()
'''

# SQLAlchemy连接池
engine = create_engine(rf"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8",
    pool_size=5,          # 连接池大小、连接数
    pool_timeout=30,      # 池中没有线程最多等待的时间，超时则报错，默认30s
    max_overflow=0,       # 超过连接池大小外最多允许创建的连接数
    pool_recycle=-1       # 多久之后对线程池中的线程进行一次连接的回收（重置），-1为不回收
)

# Pandas-SQL读写API见：Pandas/数据分析/Pandas_IO.py/4、SQL查询


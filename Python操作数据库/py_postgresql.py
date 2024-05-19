
# Python3连接PostgreSQL：psycopg2模块

# 安装：pip install psycopg2
import psycopg2

# 1）创建连接
cn = psycopg2.connect(
    host='localhost',
    port=5432,
    user='root',
    password='123456',
    database='py_pgsql'
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
print(cursor.fetchmany(3))     # 获取多个结果
result = cursor.fetchall()     # 获取所有结果
for row in result:
    print(row)                 # 以元组形式显示一行数据

# 释放资源
cursor.close()
cn.close()

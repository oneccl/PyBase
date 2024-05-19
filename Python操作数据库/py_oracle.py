
# Python3连接Oracle：cx_Oracle模块

# 安装驱动：pip install cx_Oracle
import cx_Oracle

# Python连接Oracle服务器

# 方式1：
# 监听Oracle
dsn = cx_Oracle.makedsn('host', 'port', 'db_name')
# 建立连接
cn = cx_Oracle.connect('user', 'password')

# 方式2：
cn = cx_Oracle.connect('user', 'password', f"{'host'}:{'port'}/{'db_name'}")

# 获取游标
cursor = cn.cursor()

# 执行SQL
cursor.execute('sql')

# 获取一条记录
row = cursor.fetchone()
# 获取所有记录
rows = cursor.fetchall()

# 关闭游标和连接
cursor.close()
cn.close()


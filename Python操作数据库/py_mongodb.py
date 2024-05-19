
# Python3连接MongoDB：pymongo模块

# 安装：pip install pymongo
import pymongo

# 1、创建数据库
"""
创建数据库需要使用MongoClient对象，并且指定连接的URL地址和要创建的数据库名
注意: 在MongoDB中，数据库只有在插入数据后才会真正创建
"""
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["py_mongodb"]

# 获取所有数据库
print(client.list_database_names())

# 2、创建集合
'''
注意：在MongoDB中，集合只有在插入数据后才会真正创建
'''
col = db["sites"]
# 获取所有集合
print(db.list_collection_names())

# 3、CRUD
# 3.1、插入文档
# 1）插入集合：insert_one()返回InsertOneResult对象，其包含inserted_id属性，它是插入文档的id值
d = {"name": "Tom", "age": 18}
doc_id = col.insert_one(d)
print(doc_id)
print(doc_id.inserted_id)

# 2）插入多个文档：insert_many()返回InsertManyResult对象，其包含inserted_ids属性，保存了所有插入文档的id值
ds = [{"name": "Tom", "age": 18}, {"name": "Jerry", "age": 18}]
doc_ids = col.insert_many(ds)
print(doc_ids.inserted_ids)      # 返回插入的所有文档对应的id值

# 3）插入指定id的多个文档
ds1 = [{"_id": 1, "name": "Tom", "age": 18}, {"_id": 2, "name": "Jerry", "age": 18}]
doc_ids = col.insert_many(ds1)
print(doc_ids.inserted_ids)

# 3.2、修改文档
'''
update_one()默认修改匹配到的第一条记录，如果要修改所有匹配到的记录，可以使用update_many()
'''
# 将匹配到的第一条的age值由18改为19
query = {"age": 18}
new_val = {"$set": {"age": 19}}
col.update_one(query, new_val)
# 获取修改后的集合
for x in col.find():
    print(x)

# 将匹配到的以T开头的name字段对应的age值由18改为19
query = {"name": {"$regex": "^T"}}
new_val = {"$set": {"age": 19}}
col.update_many(query, new_val)
# 获取修改后的集合
for x in col.find():
    print(x)

# 3.3、查询文档
# 1）查询集合中第一条数据：find_one()
print(col.find_one())

# 2）查询集合所有数据：find()
for x in col.find():
    print(x)
# limit()用于指定返回数据的条数
for x in col.find().limit(3):
    print(x)

# 3）查询指定字段的数据
# 除了_id，不能在一个对象中同时指定0和1，如果设置了一个字段为0，则其他都为1，反之亦然
for x in col.find({}, {"_id": 0, "name": 1, "age": 1}):
    print(x)

# 4）根据指定条件查询
query = {"name": "Tom"}
for x in col.find(query):
    print(x)          # 查找name字段为"Tom"的数据

# 5）高级查询
query = {"name": {"$gt": "T"}}
for x in col.find(query):
    print(x)          # 查找name字段中第一个字母ASCII值大于"H"的数据

query = {"name": {"$regex": "^T"}}
for x in col.find(query):
    print(x)          # 查找name字段中匹配正则表达式的数据

# 3.4、删除文档
# 1）删除一个文档：delete_one()
query = {"name": "Tom"}
col.delete_one(query)
for x in col.find():
    print(x)

# 2）删除多个文档：delete_many()
query = {"name": {"$regex": "^T"}}
x = col.delete_many(query)
print(x.deleted_count)       # 已删除数量

# 3）删除所有文档
x = col.delete_many({})
print(x.deleted_count)       # 已删除数量

# 4）删除集合：drop()
print(col.drop())            # 删除成功返回True，删除失败(集合不存在)返回False


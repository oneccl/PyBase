"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/20
Time: 22:31
Description:
"""

# 3、ConfigParser模块

# configparser是Python的标准库之一，主要用来解析.config和.ini配置文件
# config配置文件由两部分组成：sections和items
# sections用来区分不同的配置块，[]中为section；items是sections下面的键值，可以使用=或:分隔

# 例如test.config文件：
'''
[lang]
name=中文简体

[mysql]
host=localhost
port=3306
user:root
password:123456
'''
# 1）读取、解析
from configparser import ConfigParser

# 初始化
parser = ConfigParser()
parser.read(r'C:\Users\cc\Desktop\test.config', encoding='utf-8')
# 获取所有sections
print(parser.sections())       # ['lang', 'mysql']
# 获取指定section的items
print(parser.items('mysql'))   # [('host', 'localhost'), ('port', '3306'), ('user', 'root'), ('password', '123456')]
# 获取指定section对应K的V
print(parser.get('mysql', 'port'))            # 3306
# 获取第n个section的所有K
print(parser.options(parser.sections()[1]))   # ['host', 'port', 'user', 'password']

# 2）写入
parser['logging'] = {
    "level": '2',
    "path": "/root"
}
# 若是同一个ConfigParser对象，则追加写入
with open(r'C:\Users\cc\Desktop\test.config', 'w') as conf:
    parser.write(conf)

# 3）修改
parser.set('mysql', 'user', 'blue')
print(parser.get('mysql', 'user'))      # blue

# 4、ElementTree模块

# ElementTree是Python处理XML文件的内置类，用于解析、查找和修改XML，ElementTree可以将整个XML文件解析成树形结构

# 单个Element的XML对应格式：
'''
<xxx attr="xxx">xxx</xxx>
 tag attrib     text
tag：XML标签，str对象
attrib：XML属性，dict对象
text：XML数据内容
'''
# 例如test.xml文件：
'''
<?xml version="1.0" encoding="utf-8"?>
<dev_info id="netmiko_inventory">
   <R1 type="cisco">
       <device_type>cisco_ios</device_type>
       <username>admin</username>
       <password>cisco</password>
       <ip>192.168.47.10</ip>
   </R1>
   <SW3 type="huawei">
       <device_type>huawei_vrpv8</device_type>
       <username>admin</username>
       <password>huawei</password>
       <ip>192.168.47.30</ip>
   </SW3>
</dev_info>
'''
# 1）读取、解析
from xml.etree import ElementTree as ET

# 读取XML文件
tree = ET.parse(r'C:\Users\cc\Desktop\test.xml')
# 获取根信息
root = tree.getroot()
print(root.tag)        # dev_info
print(root.attrib)     # dev_info
print(root.text)
# 获取root的child层信息
for child in root:
    print(child.tag, child.attrib, child.text)
'''
R1 {'type': 'cisco'}
SW3 {'type': 'huawei'}
'''
# ElementTree查找
'''
iter(tag=None)：遍历Element的child，可以指定tag精确查找
findall(match)：查找当前元素tag或path能匹配的child节点
find(match)：查找当前元素tag或path能匹配的第一个child节点
get(key,default=None)：获取元素指定key对应的attrib，如果没有attrib，返回default
'''
for child in root.iter('password'):
    print(child.tag, child.attrib, child.text)
'''
password {} cisco
password {} huawei
'''
for child in root.findall('R1'):
    print(child.find('password').text)
'''
cisco
'''

# 2）修改、写入、删除

# 方法汇总：
'''
Element.text：修改数据内容
Element.remove(child)：删除节点
Element.set(attrib,new_text)：添加或修改属性attrib
Element.append(child)：添加新的child节点
'''
# 修改完成后使用ElementTree.write()方法写入保存

# 修改R1的ip
for child in root.iter('R1'):
    child.find('ip').text = str('192.168.47.1')

tree.write(r'C:\Users\cc\Desktop\test.xml')

# 删除SW3的标签
for child in root.findall('SW3'):
    root.remove(child)

tree.write(r'C:\Users\cc\Desktop\test.xml')



# BeautifulSoup、lxml与XPath

# 1、BeautifulSoup
# BeautifulSoup模块主要用于将HTML标签转换为Python对象树(四种对象：BeautifulSoup、Tag、NavigableString、Comment)，并从对象树中提取数据
import requests
from bs4 import BeautifulSoup

# 1）BeautifulSoup对象：代表整个HTML页面
'''
实例化BeautifulSoup对象：soup = BeautifulSoup(markup: str, features: str)
markup：待解析的字符串
features：解析器，官方建议使用lxml（可解析HTML和XML文档，并且速度快，容错能力强）
'''
# 基本用法：
resp = requests.get("https://www.crummy.com/software/BeautifulSoup/", timeout=3)
content = resp.text
# print(content)

# 将HTML字符串类型解析为BeautifulSoup对象
soup = BeautifulSoup(content, 'lxml')
# soup.prettify() 格式化、标准化HTML
print(soup.prettify())

# 2）Tag对象：网页标签、网页元素对象
# -- A、节点选择器
tag_obj = soup.h1
print(type(tag_obj))
print(tag_obj)
# 2.1）Tag对象的属性
# ①、获取标签名称
print(tag_obj.name)
# ②、获取标签的属性值
print(soup.img)                   # 获取网页的第一个img标签
print(soup.img['src'])            # 获取网页元素的DOM属性值
# ③、获取标签的所有属性
print(soup.img.attrs)             # 获取网页img标签的所有属性值，以字典形式返回
# 2.2）关联选择
print(soup.p.contents)            # 获取节点元素的直接子节点，返回List类型
print(list(soup.p.children))      # 获取节点元素的直接子节点，返回生成器类型
print(list(soup.p.descendants))   # 获取节点元素的子孙节点，返回生成器类型
print(soup.a.parent)              # 获取节点元素的直接父节点
# 2.3）其它属性
# 获取兄弟节点：next_sibling、previous_sibling、next_siblings、previous_siblings

# 3）NavigableString对象：用于获取标签内部的文字内容
# 注意：如果目标标签是一个单标签，会获取到None数据
# 3.1）使用Tag对象的属性：string、text
# 3.2）使用Tag对象的方法：get_text(split分隔符, strip=True去除空格)
nav_obj = soup.h1.string
print(type(nav_obj))
print(nav_obj)

print(soup.h1.text)
print(soup.h1.get_text('&'))

# 4）Comment对象：用于获取网页的注释内容

# -- B、方法选择器：find()、find_all()
# 1）调用BeautifulSoup对象和Tag对象的find()方法，可以获取网页指定标签元素对象，方法返回查找到的第一个元素
'''
find(name,attrs,recursive,text)
name：标签名称   attrs：标签属性   recursive：默认搜索所有后代元素   text：标签内容
'''
# ①、标签名称查找
# print(soup.find(name='a'))
# ②、attrs参数查找
# print(soup.find(attrs={'id': 'cta'}))
# print(soup.find(attrs={'class': 'cta'}))
# ③、特殊参数查找
# print(soup.find(id='cta'))
# print(soup.find(class_='cta'))

# 2）调用BeautifulSoup对象和Tag对象的find_all()方法，可以获取网页指定标签元素对象，方法返回查找到的全部匹配元素
'''
find_all(name,attrs,recursive,text,limit)
limit：最多返回的匹配数量（find()可看作limit=1）
'''
# print(soup.find_all('a'))

# 3）其他方法：
# 获取兄弟节点：find_next_sibling()、find_previous_sibling()、find_next_siblings()、find_previous_siblings()

# -- C、CSS选择器
print(soup.select('ul li'))                   # 获取ul下的li节点
print(soup.select('#list2 li'))               # 获取id=list2下的li节点
print(soup.select('ul'))                      # 获取所有的ul节点
# 嵌套选择：获取属性
# for ul in soup.select('ul'):
#     print(ul.select('li'))
#     print(ul['id'])
#     print(ul.attrs['id'])

# 嵌套选择：获取文本
# for li in soup.select('li'):
#     print(li.string)
#     print(li.get_text())

# 案例：获取B站弹幕
# 弹幕接口：https://api.bilibili.com/x/v1/dm/list.so?oid=276746872

# headers = {'user-agent': 'Mozilla/5.0'}
# url = 'https://api.bilibili.com/x/v1/dm/list.so?oid=276746872'
# # 获取网页信息
# response = requests.get(url, headers=headers)
# html = response.content.decode('utf-8')
# # 打印弹幕
# soup = BeautifulSoup(html, 'lxml')
# for d in soup.find_all(name='d'):
#     print(d.string)


# 2、lxml与XPath
# 1）lxml模块主要用于创建、解析和查询XML和HTML文档
from lxml import etree

# etree.HTML()：将HTML字符串转换解析为Element对象（若HTML代码不规范，可以自动补全标签）
# etree.tostring()：将Element对象再转换解析为HTML字符串（爬虫应该以此结果作为提取数据的依据）

# Element对象的xpath()方法可以执行XPath查询，从而获取指定元素的内容

# 2）XPath(XML Path Language)是一门在HTML/XML文档中查找信息的语言，可用来在HTML/XML文档中对元素和属性进行遍历
'''
XPath节点：每个html、xml标签都是一个节点，其中最顶层的节点称为根节点
'''
# 2.1）XPath语法：基础节点选择语法
# XPath定位节点以及提取属性或文本内容的语法:
# XPath使用路径表达式来选取XML文档中的节点或节点集，这些路径表达式和我们在常规电脑文件系统中看到的表达式类似
'''
路径表达式                 描述
nodename                  选中该元素
/                         从根节点选取或元素和元素间的过渡
//                        从当前节点选择文档中的节点，而不考虑它们的位置
.                         选取当前节点
...                       选取当前节点的父节点
@                         选取属性
text()                    选取文本
'''
# 示例：
'''
获取所有的h2下的文本：//h2/text()
获取所有的a标签的href：//a/@href
获取html下head下title的文本：/html/head/title/text()
获取html下head下link标签的href：/html/head/link/@href
'''
# 2.2）XPath语法：节点修饰语法
# 可以根据标签的属性值、下标等来获取特定的节点
'''
路径表达式                                 描述
//title[@lang='eng']	                  选取lang属性值为eng的所有title元素
/store/book[1]	                          选取store子元素的第一个book元素
/store/book[last()]	                      选取store子元素的最后一个book元素
/store/book[position()>1]	              选取store下面的book元素，从第二个开始选择
//book/title[text()='text']	              选择所有book下的title元素，仅选择文本为text的title元素
/store/book[price>35.00]/title	          选取store元素中的book元素的所有title元素，且其中price元素的值必须大于35.00
'''
# 2.3）XPath语法：其他常用节点选择语法
# 可以通过通配符来选取未知的html、xml的元素
'''
通配符                 描述
*                     匹配任何元素节点
node()                匹配任何类型的节点
'''
# 示例：
'''
选取全部的标签：//*
选取全部的属性：//node()
'''

# 3）lxml与XPath结合使用

# html = etree.HTML('text')
# res_list = html.xpath('XPath表达式')
# xpath()方法返回列表的三种情况
'''
①、返回空列表：根据XPath语法规则字符串，没有匹配到任何元素
②、返回由字符串构成的列表：XPath字符串规则匹配的一定是文本内容或某属性的值
③、返回由Element对象构成的列表：XPath规则字符串匹配的是标签，列表中的Element对象可以继续进行xpath()
'''
# xpath()提取数据的方法：
'''
# 返回符合XPath表达式的所有数据列表
xpath('XPath表达式').getall() / xpath('XPath表达式').extract()
# 返回符合XPath表达式的数据列表的第一个数据
xpath('XPath表达式').get() / xpath('XPath表达式').extract_first()
'''
# 注意：extract()、extract_first()方法获取不到返回None；get()、getall()方法获取不到会Raise一个错误

# 案例：将HTML文档中每个class为item-1的li标签作为一条数据，提取a标签的文本内容及链接，组装成一个字典
text = '''
<div> 
  <ul> 
    <li class="item-1">
      <a href="link1.html">first item</a>
    </li> 
    <li class="item-1">
      <a href="link2.html">second item</a>
    </li> 
    <li class="item-0">
      <a href="link3.html">third item</a>
    </li> 
    <li class="item-1">
      <a href="link4.html">fourth item</a>
    </li> 
    <li class="item-0">
      a href="link5.html">fifth item</a>
  </ul> 
</div>
'''
# 根据li标签进行分组
html = etree.HTML(text)
li_list = html.xpath("//li[@class='item-1']")
# 在每组中进行数据提取
for li in li_list:
    item = {}
    item['href'] = li.xpath("./a/@href")[0] if len(li.xpath("./a/@href")) > 0 else None
    item['content'] = li.xpath("./a/text()")[0] if len(li.xpath("./a/text()")) > 0 else None
    print(item)


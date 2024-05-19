
# 网络爬虫（基本库）

"""
1、基本概念
网络爬虫（Web Crawler）是一种自动化程序，用于从互联网上收集信息。它通过自动访问网页并提取所需的数据，实现对大量网页的快速检索和数据抓取
网络爬虫通常使用HTTP协议来访问网页，并通过解析HTML、XML等网页内容来提取数据。爬虫可以从一个起始点（如某个特定网页）开始，然后根据链接关系自动地遍历和抓取其他相关网页
"""
'''
1.1、工作原理
a、确定起始点：选择一个或多个起始网页作为爬虫的入口点
b、发送HTTP请求：通过HTTP协议向起始网页发送请求，并获取网页的内容
c、解析网页：解析网页的内容，通常使用HTML解析器或XML解析器来提取所需的数据
d、提取链接：从解析后的网页中提取其他相关网页的链接
e、存储数据：将爬取到的数据存储到数据库、文件或其他存储介质中
f、遍历网页：根据提取到的链接继续遍历和抓取其他相关网页，重复上述步骤
网络爬虫可以根据需要进行配置，例如设置爬取的深度、限制爬取的速度，以及处理反爬机制等

1.2、分类与应用
a、网络爬虫根据不同的目标和应用可以分为多种类型，如通用爬虫、聚焦爬虫、增量爬虫等
1）通用爬虫是一种广泛应用的爬虫，可以遍历互联网上的大部分网页，并抓取数据进行索引和检索。搜索引擎的爬虫就是一种通用爬虫的例子
2）聚焦爬虫是针对特定领域或特定网站进行抓取的爬虫。它只关注特定的内容，能够更精准地抓取所需的数据
3）增量爬虫是在已有数据的基础上，只抓取新增或更新的数据。它可以根据时间戳或其他标识来判断数据的更新情况，减少重复抓取和处理的工作量
b、网络爬虫在很多领域都有应用，如搜索引擎、数据挖掘、舆情分析、价格比较等

1.3、网络（爬虫的法律与道德问题）
网络爬虫在使用过程中需要注意法律和道德问题。以下是一些常见的问题：
1）合法性：爬虫的行为必须遵守相关法律法规，尊重网站的隐私权和知识产权。不得未经授权地访问和抓取受保护的网页内容
2）访问频率：爬虫应该合理设置访问频率，避免对网站服务器造成过大负载或影响其他用户的正常访问
3）数据使用：抓取到的数据应该按照法律和道德准则进行合法和合理的使用，遵守数据保护和隐私规定
4）在使用网络爬虫时，应当遵守相关规定并尊重网站的权益和用户的隐私
'''

# 2、Python基本爬虫库
'''
2.1、requests库
requests是一个常用的第三方库，用于发送HTTP请求和处理响应。它提供了简洁而直观的API，使得发送请求和处理数据变得非常方便
'''
import requests

# 发送get请求
response = requests.get("https://www.baidu.com/")

# 获取响应内容
print(response.text)

# 发送post请求
data = {'key': 'value'}
requests.post("https://www.example.com", data=data)

# 获取响应状态码
print(response.status_code)

'''
2.2、urllib库
urllib是Python的标准库之一，用于处理URL相关的操作。它包含多个模块，如urllib.request用于发送HTTP请求，urllib.parse用于处理URL解析，urllib.error用于处理异常等
'''
from urllib import request

# 发送get请求
resp = request.urlopen("https://www.baidu.com/")

# 获取响应内容
print(resp.read().decode('utf-8'))

# 发送post请求
data = b'key=value'
resp = request.urlopen("https://www.example.com", data=data)

# 获取响应状态码
print(resp.status)

'''
2.3、BeautifulSoup库
BeautifulSoup是一个用于解析HTML和XML文档的库，它可以将复杂的文档转换为易于遍历和搜索的Python对象
'''
from bs4 import BeautifulSoup
html_doc = '''
<html>
<head>
    <title>Example</title>
</head>
<body>
    <div class="content">
        <h1>Hello, World!</h1>
        <p>This is an example.</p>
    </div>
</body>
</html>
'''
# 通过BeautifulSoup类的构造函数可以将HTML文档转换为BeautifulSoup对象
soup = BeautifulSoup(html_doc, 'html.parser')

# 获取标题
title = soup.title.string
print(title)               # Example

# 获取内容：find()方法：可以搜索指定的元素；get_text()方法：可以获取元素的文本内容
content = soup.find('div', class_='content')
print(content.get_text())  # Hello, World!
                           # This is an example.

'''
2.4、lxml库、XPath库
lxml是一个高性能的XML和HTML处理库，它基于C语言库libxml2和libxslt，提供了方便的API用于解析和处理XML和HTML文档
'''
from lxml import etree

# 将HTML文档转换为ElementTree对象
tree = etree.HTML(html_doc)

# 获取标题
title = tree.xpath('//title/text()')[0]
print(title)

# 获取内容：xpath()方法：可以执行XPath查询，从而获取指定元素的内容
content = tree.xpath('//div[@class="content"]/text()')
print(''.join(content).strip())


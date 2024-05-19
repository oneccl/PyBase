
# urllib模块是Python标准库，用于处理URL相关的操作
# urllib库包含四个子模块：
"""
urllib.request：请求模块，用于打开和读取URL
urllib.error：异常处理模块，捕获urllib.error抛出异常
urllib.parse：URL解析，爬虫程序中用于处理URL地址
urllib.robotparser：解析robots.txt文件（一个遵循Robots Exclusion Protocol（机器人排除协议）的文本文件），判断目标站点哪些内容可爬，哪些不可以爬
"""
# 1）urllib.request模块
import urllib.request

# urlopen(url,data,timeout) 发送Get、Post请求
with urllib.request.urlopen('https://www.example.com') as resp:
    # getcode() 获取网页状态码
    print(resp.getcode())
    # read() 获取服务器响应的内容，读取整个网页数据
    content = resp.read().decode('utf-8')
    print(content)

# HTTPResposne响应对象的其它成员
'''
getheaders()：获取请求头内容
getheader(name)：获取指定请求头
readline()：读取网页一行数据
readlines()：读取网页多行数据
status：获取响应状态码
'''

# Request()类 发送指定类型请求
'''
urllib.request.Request(url,data,headers,method)
'''
import urllib.parse

# 转换数据类型为bytes（data参数类型必须为bytes类型）
data = urllib.parse.urlencode({"key": "value"}).encode()
headers = {"User-Agent": "Mozilla/5.0"}
req = urllib.request.Request(url='https://www.example.com', data=data, method='POST', headers=headers)
with urllib.request.urlopen(req) as resp:
    # getcode() 获取网页状态码
    print(resp.status)
    # read() 获取服务器响应的内容，读取整个网页数据
    content = resp.read().decode('utf-8')
    print(content)

# 2）urllib.parse模块

# 解析URL: urlparse(url,scheme协议类型,allow_fragments=True是否忽略URL中的fragment部分)
# 标准的URL格式：scheme://netloc/path;params?query#fragment
'''
scheme：URL协议
netloc：域名和端口
path：路径
params：最后一个路径元素参数
query：查询字符串
fragment：片段标志
'''
from urllib.parse import urlparse

url = 'https://www.example.com/path?query=hello#fragment'
parsed_url = urlparse(url)

print(parsed_url.scheme)       # https
print(parsed_url.netloc)       # www.example.com
print(parsed_url.path)         # /path
print(parsed_url.query)        # query=hello
print(parsed_url.fragment)     # fragment

# 其它方法：
'''
urlunparse()：构建URL，与urlparse()方法逻辑相反
urljoin()：方法用于拼接链接
urlencode()：格式化URL请求参数
quote()：编码URL特殊字符，尤其是转换中文字符
unquote()：解码URL特殊字符
'''

# 3）urllib.error模块

import urllib.error

# 异常类：
# error.URLError：OSError的一个子类，用于处理URL相关的错误，如无法连接到服务器、网络问题等
# error.HTTPError：URLError的一个子类，用于处理HTTP相关的错误，如页面不存在（404）、权限问题等

# 异常类对象提供了一些属性来获取更详细的错误信息：
# e.code：响应的状态码；e.reason：获取原因字符串；e.headers：获取响应的头部信息

# 4）urllib.robotparser模块


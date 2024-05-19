# requests第三方库
import requests

# 1、发送请求 Request

# 1.1、7个主要方法：
'''
1）requests.request(method, url, params, data, headers, cookies)
  - method: 请求方式，对应以下各方法  
  - url: 页面链接  
  - params: url参数，字典或字节流类型 
  - data: 请求内容、请求体，字典、字节序列或文件对象类型
  - headers: HTTP自定义请求头，字典类型
  - cookies: 请求中的cookie，字典类型
  - json: 请求内容、请求体，JSON格式的数据
  - files: 请求内容、请求体，用于传输文件
  - timeout: 设置请求超时时间，单位秒
  - allow_redirects: 重定向开关，默认为True
  - proxies: 设置访问代理服务器，可以增加登录认证，字典类型
2）requests.get()         # 获取HTML网页的主要方法，对应HTTP的GET
3）requests.head()        # 获取网页头信息的方法，对应HTTP的HEAD（HTTP 头部本质上是一个传递额外重要信息的键值对）
4）requests.post()        # 向HTML网页提交POST请求，对应HTTP的POST
5）requests.put()         # 向HTML网页提交PUT请求，对应HTTP的PUT
6）requests.delete()      # 向HTML网页提交删除请求，对应HTTP的DELETE
7）requests.patch()       # 向HTML网页提交局部修改请求，对应HTTP的PATCH
'''

# 1.2、get请求

# 1）基本用法
response = requests.get('https://httpbin.org/get')
print(response.text)

# 2）url参数
params = {"k1": "v1", "k2": "v2"}
response = requests.get('https://httpbin.org/get', params=params)
print(response.text)

# 3）超时参数：如果一个请求在给定时间内没有返回结果，就抛出异常
response = requests.get('https://httpbin.org/get', timeout=3)
print(response.text)

# 4）自定义headers：反爬第一阶段常用套路，headers参数中可携带cookie
headers = {"Content-Type": "application/octet-stream", "cookie": 'cookie'}
response = requests.get('https://httpbin.org/get', headers=headers)
print(response.text)

# 1.3、post请求

# 5）请求内容、请求体
body = {"name": "Tom", "age": 18}
response = requests.post('https://httpbin.org/post', data=body)
print(response.text)

# post请求其他使用和get相同

# 2、接收响应 Response

# 2.1、Response对象的属性和方法
print(response.status_code)            # HTTP请求返回状态码，200:连接成功，404或其他:连接失败
print(response.text)                   # HTTP请求响应内容的字符串格式，即url对应的页面内容
print(response.url)                    # 获取响应的完整URL
print(response.encoding)               # HTTP header中响应内容的编码格式
print(response.apparent_encoding)      # 从响应内容中分析其编码格式
print(response.headers)                # 获取响应报文头
print(response.cookies)                # 获取cookie
print(response.content)                # HTTP请求响应内容的二进制字节码格式
print(response.history)                # 获取请求历史
print(response.json())                 # 将响应的JSON字符串反序列化为Python对象

# 2.2、中文乱码问题
# 通过对response.content进行decode，来解决中文乱码问题
response.content.decode()              # 默认utf-8
response.content.decode("GBK")         # 方式1：指定编码GBK
response.encoding = 'GBK'              # 方式2：指定编码GBK

# 3、requests库的常用异常
'''
requests.ConnectionError                # 网络连接异常
requests.HTTPError                      # HTTP错误异常
requests.URLRequired                    # URL缺失异常
requests.TooManyRedirects               # 超过最大重定向次数，产生重定向异常
requests.ConnectTimeout                 # 连接远程服务器超时异常
requests.Timeout                        # 请求URL超时异常
'''
# 异常处理常用方法 response.raise_for_status()
# 该方法判断response.status_code是否等于200（是否请求成功），不需要额外if语句

# 4、通用框架
def get_html_text(url):
    try:
        # 发送请求
        resp = requests.get(url, timeout=30)
        # 若响应状态码不是200，则触发异常
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        # 获取响应内容
        return resp.text
    except:
        print("爬取失败！")


# 5、requests库的高级操作

# 1）处理获取到的cookie
# 方式1：requests.utils.dict_from_cookiejar() 将cookies转换为字典
dict_cookie = requests.utils.dict_from_cookiejar(response.cookies)
print(dict_cookie)
# 方式2：遍历
dict_cookie = {name: value for name, value in response.cookies.items()}
print(dict_cookie)

# 2）文件上传（文件类型会自动进行处理）
files = {"file": open('../upload/a.jpg', 'rb')}
def upload(url, files):
    resp = requests.post(url, files=files)
    print(resp.text)

# 3）身份认证
from requests.auth import HTTPBasicAuth

resp1 = requests.get("https://httpbin.org/get", auth=HTTPBasicAuth('用户名', '密码'))
print(resp1.status_code)
resp2 = requests.get("https://httpbin.org/get", auth=('用户名', '密码'))    # 简写
print(resp2.status_code)

# 4）会话维持：通过get()和post()请求登录网站，相当打开了两个浏览器
# 方式1：可以在两次请求时设置一样的Cookies
# 方式2：通过Session类
session = requests.Session()
# 打开一个会话，首先并向cookies中设置参数number=123
resp1 = session.get("https://httpbin.org/ciikies/set/number/123")
# 第二次请求获取cookies内容
resp2 = session.get("https://httpbin.org/ciikies/")
print(resp2.text)
session.close()

# 5）SSL证书验证
# 当发送HTTP请求时，客户端会检查SSL证书，若该网站的的证书没有被CA机构信任，程序将抛出一个异常，提示SSL证书验证错误
# 可以通过requests的verify参数控制是否检查SSL证书，当设置verify=False时，程序运行会产生警告
# 5.1）可以使用urllib3.disable_warnings()屏蔽警告，也可以通过捕获警告到日志的方式忽略警告
from requests.packages import urllib3

urllib3.disable_warnings()
resp = requests.get("https://www.12306.cn", verify=False)
print(resp.status_code)

# 5.2）也可以指定本地证书作为客户端证书，它可以是单个文件(包含秘钥和证书)，也可以是包含两个文件的元组
# key文件（key必须是解密状态，加密状态的key是不支持的）和crt文件
import logging

logging.captureWarnings(True)
resp = requests.get('https://www.12306.cn', cert('../server.crt', '../key'))
print(resp.status_code)

# 6）代理设置
# 对于某些网站，在测试几次请求时可以正常获取内容；但当大规模、频繁地爬取，网站可能会弹出验证码或跳转到登录页面
# 甚至直接封禁客户端IP，导致一定时间内无法访问。可以通过proxies参数使用代理解决这种问题
# 6.1）HTTP代理
proxies = {'http': 'http://161.35.4.201:80', 'https': 'https://161.35.4.201:80'}
try:
    resp = requests.get("https://httpbin.org/get", proxies=proxies)
    print(resp.text)
except requests.exceptions.ConnectionError as e:
    print(f"Error: {e.args}")

# 通过结果中origin可以发现，使用的是代理服务器进行访问的

# 6.2）如果代理需要使用HTTPBasicAuth，可以使用类似http://user:password@host:port这样的语法来设置代理
proxies = {
  "http": "http://user:password@161.35.4.201:80"
}
resp = requests.get("https://www.taobao.com", proxies=proxies)
print(resp.text)

# 6.3）SOCKS协议代理（pip install 'requests[socks]'）
proxies = {
  'http': 'socks5://user:password@host:port',
  'https': 'socks5://user:password@host:port'
}
resp = requests.get('https://www.taobao.com', proxies=proxies)
print(resp.text)

# 7）Prepared Request
# urllib库中，发送请求需要设置请求头时可以通过Request对象来表示，requests库也提供了与之类似的类Prepared Request
# 优点：可以利用Request将请求当作独立的对象，方便进行队列调度，可以用它来构造Request队列
from requests import Request, Session

data = {'name': 'Alice'}
headers = {'User-Agent': '请求身份信息(一般使用浏览器的)', 'Accept': 'application/json, charset=utf-8'}
session = Session()
# 使用url、data和headers参数构造一个Request对象
req = Request('POST', url='https://httpbin.org/post', data=data, headers=headers)
# 使用Session的prepare_request()方法将其转换为一个Prepared Request对象
prepared = session.prepare_request(req)
# 使用Session的send()方法发送请求
resp = session.send(prepared)
print(resp.text)

# 6、分析案例
'''
A、通过抓包分析步骤：开发者工具->Network(网络)->Headers(标头)->Fetch/XHR->
Request URL(请求URL（包含请求参数）)
Request Method(请求方法)
Response Headers响应标头->Content-Type(返回数据格式)
[负载->]Form Data(表单数据（包含请求内容、请求体）)

B、headers: 开发者工具->Network(网络)->Headers(标头)->Request Headers(请求标头)->User-Agent
'''

# 6.1、案例1：爬取百度翻译：https://fanyi.baidu.com/?aldtype=16047#en/zh
import json

url = 'https://fanyi.baidu.com/sug'
text = input("请输入需要翻译的单词: ")
data = {"kw": text}
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.post(url=url, data=data, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    content = response.json()
    # json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置成False
    res = json.dumps(content, ensure_ascii=False)
    # print(res)
    result = json.loads(res)
    print(result['data'][0]['v'].split(";")[0].split(". ")[1])
except:
    print("爬取失败！")

# 6.2、案例2：爬取豆瓣电影排行榜：https://movie.douban.com/chart
import json

url = 'https://movie.douban.com/j/chart/top_list'
params = {
    'type': '24',
    'interval_id': '100:90',
    'action': '',
    'start': '0',
    'limit': '20'
}
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.get(url=url, params=params, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    content = response.json()
    # json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置成False
    res = json.dumps(content, ensure_ascii=False)
    print(res)
except:
    print("爬取失败！")

# 6.3、案例3：爬取肯德基指定省市有多少家餐厅：http://www.kfc.com.cn/kfccda/index.aspx
import json

url = 'http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=keyword'
text = input("请输入需要查询的省市: ")
data = {
    "cname": "",
    "pid": "",
    "keyword": text,
    "pageIndex": 1,
    "pageSize": 10
}
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.post(url=url, data=data, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    content = response.json()
    # json.dumps()会默认将中文转化成ASCII编码格式，因此需要手动设置成False
    res = json.dumps(content, ensure_ascii=False)
    # print(res)
    result = json.loads(res)
    print(result['Table'][0]['rowcount'])
except:
    print("爬取失败！")


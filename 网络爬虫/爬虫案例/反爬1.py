"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/10/5
Time: 19:01
Description:
"""

# 反爬虫机制与反爬虫技术

# 1）User-Agent伪装
'''
User-Agent能够通过服务器识别出用户的操作系统及版本、CPU类型、浏览器类型及版本等。一些网站会根据请求头中的Referer和User-Agent信息来判断请求的合法性。部分网站会设置User-Agent白名单
只有在白名单范围内的请求才可以正常访问，因此，在我们爬虫时，需要设置User-Agent伪装成一个浏览器HTTP请求，通过修改User-Agent，可以模拟不同的浏览器或设备发送请求，从而绕过一些简单的反爬虫机制
Referer是HTTP请求头中的一个字段，用于指示请求的来源页面。当Referer为空或不符合预期值时，网站可能会拒绝请求或返回错误的数据。Referer一般指定为爬取网站的主页地址
'''
# 2）代理IP
'''
一些网站通常会根据IP地址来判断请求的合法性，如果同一个IP地址频繁请求，就会被认为是爬虫。使用IP代理可以隐藏真实的IP地址，轮流使用多个IP地址发送请求，可以增加爬虫的隐匿性
代理IP是指通过中间服务器转发网络请求的技术。在爬虫中，使用代理IP可以隐藏真实的访问源，防止被目标网站封禁或限制访问
代理分为正向代理和反向代理。正向代理是由客户端主动使用代理服务器来访问目标网站，反向代理是目标网站使用代理服务器来处理客户端的请求
'''
# 代理IP优缺点：
'''
优点：
1）隐藏真实的访问源，保护个人或机构的隐私和安全
2）绕过目标网站的访问限制，如IP封禁、地区限制等
3）分散访问压力，提高爬取效率和稳定性
4）收集不同地区或代理服务器上的数据，用于数据分析和对比
缺点：
1）代理IP的质量参差不齐，有些代理服务器可能不稳定、速度慢或存在安全风险
2）一些目标网站会检测和封禁常用的代理IP，需要不断更换和验证代理IP的可用性
3）使用代理IP可能增加网络请求的延迟和复杂性，需要合理配置和调整爬虫程序
4）使用代理IP需要遵守相关法律法规和目标网站的使用规则，不得进行非法活动或滥用代理IP服务
'''
# 亮数据代理IP：https://www.bright.cn/locations
# 3）请求频率控制
'''
频繁的请求会给网站带来较大的负担，并影响网站的正常运行，因此，网站通常会设置请求频率限制。Python中的time库可以用来控制请求的时间间隔，避免过于频繁的请求
'''
# 4）动态页面处理
'''
一些网站为了防止爬虫，使用了JavaScript来动态生成页面内容，这对于爬虫来说是一个挑战。Python中的Selenium库可以模拟浏览器的行为，执行JavaScript代码，从而获取动态生成的内容
例如在进行数据采集时，很多网站需要进行登录才能获取到目标数据，这时可以使用Selenium库进行模拟登录进行处理
'''
# 5）验证码识别
'''
一些网站为了防止爬虫，会在登录或提交表单时添加验证码。随着反爬的不断发展，逐渐出现了更多复杂的验证码，例如：内容验证码、滑动验证码、图片拼接验证码等
Python提供了一些强大的图像处理库，例如Pillow、OpenCV等，可以用来自动识别验证码，从而实现自动化爬取
'''

# 案例：豆瓣电影Top250爬取
# 1）2）3）反爬虫技术的使用

# 爬取目标
'''
豆瓣电影Top250排行榜：爬取字段：排名、电影名、评分、评价人数、制片国家、电影类型、上映时间、主演、影片链接
'''
# 豆瓣：https://www.douban.com/
# 豆瓣电影：https://movie.douban.com/
# 豆瓣电影Top250：https://movie.douban.com/top250

# 模块导入
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree
import re
import time

# 翻页分析
'''
第1页：https://movie.douban.com/top250?start=0&filter=
第2页：https://movie.douban.com/top250?start=25&filter=
第3页：https://movie.douban.com/top250?start=50&filter=
......
'''
# 构造每页的网页链接
urls = [rf'https://movie.douban.com/top250?start={str(i * 25)}&filter=' for i in range(10)]

# 发送请求
def get_html_str(url: str):
    # 请求头模拟浏览器
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.douban.com'}
    # 代理IP
    proxies = {"http": "http://183.134.17.12:9181"}
    # 发送请求
    resp = requests.get(url, headers=headers, proxies=proxies)
    # 获取网页源代码
    html_str = resp.content.decode()
    return html_str


# 数据提取与解析
# 分析网页数据

# 1）方式1：使用BeautifulSoup库
def bs4_html_parser(content: str):
    soup = BeautifulSoup(content, 'lxml')
    data = []
    for tag in soup.find_all(attrs={"class": "item"}):
        # 排名
        rank = tag.find("em").get_text()
        # 电影名
        title = tag.find_all(attrs={"class": "title"})[0].get_text()
        # 评分、评价人数
        text = tag.find(attrs={"class": "star"}).get_text().replace('\n', ' ').strip()
        score = text.split('  ')[0]
        numbers = text.split('  ')[1].replace('人评价', '')
        # 导演、主演
        s = tag.find(attrs={"class": "bd"}).p.get_text().strip()
        ls = [line.strip() for line in s.split('\n')]
        ls1 = ls[0].split('\xa0'*3)
        director = [e.strip() for e in ls1[0].replace('导演:', '').strip().split('/')]
        performer = [e.replace('...', '').strip() for e in ls1[1].replace('主演:', '').strip().split('/') if e.strip() != '...']
        # 年份、制片国家、影片类型
        ls2 = [re.sub('\xa0', '', e) for e in ls[1].split('/')]
        year = ls2[0]
        country = ls2[1].split(' ')
        type = ls2[2].split(' ')
        # 链接
        url = tag.find(attrs={"class": "hd"}).a.attrs['href']
        print({'排名': rank, '电影名': title, '评分': score, '评价人数': numbers, '导演': director, '主演': performer, '上映年份': year, '制片国家': country, '影片类型': type, '影片链接': url})
        data.append(
            {'排名': rank, '电影名': title, '评分': score, '评价人数': numbers, '导演': director, '主演': performer, '上映年份': year, '制片国家': country, '影片类型': type, '影片链接': url}
        )
    return pd.DataFrame(data)

# 2）方式2：使用lxml与XPath库
def lxml_html_parser(content: str):
    # 将HTML字符串转换为Element对象
    html = etree.HTML(content)
    # 使用XPath获取所有li标签
    li_list = html.xpath("//ol[@class='grid_view']/li")
    # print(len(li_list))
    data = []
    # 遍历li标签，从中提取数据信息
    for li in li_list:
        # 排名
        rank = li.xpath(".//div[@class='pic']/em/text()")[0]
        # 电影名
        title = li.xpath(".//div[@class='hd']/a/span[1]/text()")[0]
        # 评分
        score = li.xpath(".//span[@class='rating_num']/text()")[0]
        # 评价人数
        numbers = li.xpath(".//div[@class='star']/span[4]/text()")[0]
        numbers = numbers.replace("人评价", "")
        # 导演、主演
        s1 = li.xpath(".//div[@class='bd']/p[1]//text()")[0].strip()
        # print(s1)
        # 使用正则提取导演
        try:
            director = re.sub('\xa0', '', re.findall("导演: (.*?)主演", s1)[0])
            director = [e.strip() for e in director.split('/')]
        except:
            director = None
        # 使用正则提取主演
        try:
            performer = re.sub('\xa0', '', re.findall("主演: (.*)", s1)[0])
            performer = [e.replace('...', '').strip() for e in performer.split('/') if e.strip() != '...']
        except:
            performer = None
        # 上映时间、制片国家、电影类型
        s2 = li.xpath(".//div[@class='bd']/p[1]//text()")[1].strip()
        # print(s2)
        try:
            ls = [re.sub('\xa0', '', e) for e in s2.split('/')]
            # 年份
            year = ls[0]
            # 制片国家
            country = ls[1].split(' ')
            # 影片类型
            type = ls[2].split(' ')
        except:
            year = None
            country = None
            type = None
        # 链接
        url = li.xpath(".//div[@class='hd']/a/@href")[0]
        # print({'排名': rank, '电影名': title, '评分': score, '评价人数': numbers, '导演': director, '主演': performer, '上映年份': year, '制片国家': country, '影片类型': type, '影片链接': url})
        data.append(
            {'排名': rank, '电影名': title, '评分': score, '评价人数': numbers, '导演': director, '主演': performer, '上映年份': year, '制片国家': country, '影片类型': type, '影片链接': url}
        )
    return pd.DataFrame(data)


# 案例调试
html_str = get_html_str("https://movie.douban.com/top250?start=0&filter=")
data = lxml_html_parser(html_str)
# print(data.to_string())

# 保存数据测试
# data.to_csv(r"C:\Users\cc\Desktop\豆瓣电影Top250排行.csv", index=False)


# 主函数
if __name__ == '__main__':
    data_res = pd.DataFrame()
    for url in urls:
        html = get_html_str(url)
        data = lxml_html_parser(html)
        data_res = pd.concat([data_res, data], ignore_index=True)
        time.sleep(5)
    print(data_res.to_string())
    data_res.to_csv(r"C:\Users\cc\Desktop\豆瓣电影Top250排行.csv", index=False, encoding='utf-8')


# 电影数据网站：
# 艺恩娱数：https://ys.endata.cn/DataMarket/Index


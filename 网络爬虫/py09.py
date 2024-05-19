
# 网络爬虫（爬虫框架Scrapy）

"""
1、Scrapy框架
Scrapy是一个功能强大的Python开源爬虫框架，用于快速、高效地抓取和提取结构化数据。它提供了一套完整的工具和组件，使得构建爬虫变得简单和灵活
Scrapy框架采用异步的方式处理请求和响应，支持多线程和分布式爬取。它提供了方便的API和命令行工具，可以配置爬虫规则、处理数据和导出结果
"""
'''
2、安装与配置
1）安装Scrapy：pip install scrapy
2）验证Scrapy的安装：scrapy --version
'''

'''
3、Scrapy的基本使用
使用Scrapy框架编写爬虫主要包括定义爬虫类、编写爬虫规则和处理数据
'''
'''
4、Scrapy的数据存储与导出
Scrapy提供了多种方式来存储和导出爬取的数据，包括保存为JSON、CSV、XML等格式；存储到数据库中，或自定义存储管道
'''
import scrapy
from scrapy.exporters import CsvItemExporter   # Scrapy的数据存储与导出

class MySpider(scrapy.Spider):
    name = 'myspider'
    # 设置起始URL
    start_urls = ['https://www.baidu.com']

    # 实现parse()方法来处理响应数据
    def parse(self, response):
        # 处理响应数据
        title = response.xpath('//title/text()').get()
        print(title)

        # 导出数据：导入CsvItemExporter类和设置相关配置，将爬取的数据保存为名为data.csv的CSV文件
        exporter = CsvItemExporter(open('data.csv', 'wb'))
        exporter.start_exporting()
        exporter.export_item({'title: ': title})
        exporter.finish_exporting()

# 运行爬虫：通过导入CrawlerProcess类和设置相关配置，可以创建爬虫进程并运行爬虫
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'
    })

    process.crawl(MySpider)
    process.start()


# 爬虫实践与案例

'''
1、爬取网页数据（使用requests库和BeautifulSoup库爬取网页数据）
爬取网页数据是最常见的爬虫应用之一。通过发送HTTP请求，解析网页内容，提取所需的数据
'''
import requests
from bs4 import BeautifulSoup

# 发送HTTP请求
response = requests.get('https://www.example.com')

# 解析网页内容
soup = BeautifulSoup(response.text, 'html.parser')

# 提取所需的数据
title = soup.title.string
print(title)

'''
2、爬取图片数据（使用requests库下载图片数据）
爬取图片数据是一种常见的应用场景，例如爬取图片网站上的图片数据
'''
import requests

# 发送HTTP请求下载图片
image_url = 'https://www.example.com/image.jpg'
response = requests.get(image_url)

# 保存图片数据到本地文件
with open('image.jpg', 'wb') as f:
    f.write(response.content)

'''
3、爬取视频数据（使用requests库下载视频数据）
爬取视频数据是一种常见的应用场景，例如从视频分享网站上爬取视频数据
'''
import requests

# 发送HTTP请求下载视频
video_url = 'https://www.example.com/video.mp4'
response = requests.get(video_url)

# 保存视频数据到本地文件
with open('video.mp4', 'wb') as f:
    f.write(response.content)

'''
4、爬取社交媒体数据（使用Twitter API爬取推文数据）
爬取社交媒体数据是一种常见的应用场景，例如爬取Twitter、Facebook等社交媒体平台上的数据
'''
import tweepy

# Twitter API认证
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# 创建API对象
api = tweepy.API(auth)

# 获取指定用户的最新推文
tweets = api.user_timeline(screen_name='twitter', count=10)

# 打印推文内容
for tweet in tweets:
    print(tweet.text)


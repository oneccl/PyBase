"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2024/7/23
Time: 22:03
Description:
"""
# 爬取大众点评指定城市【美食页】

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import random
import time

# from urllib.parse import quote, unquote
#
# # URL加密与解密（中文乱码）
# ms = "美食"
# print(quote(ms))  # %E7%BE%8E%E9%A3%9F
# print(unquote("%E7%BE%8E%E9%A3%9F"))  # 美食


# 免费代理IP
proxies = [
    {'http': '183.95.80.102:8080'},
    {'http': '123.160.31.71:8080'},
    {'http': '115.231.128.79:8080'},
    {'http': '166.111.77.32:80'},
    {'http': '43.240.138.31:8080'},
    {'http': '218.201.98.196:3128'},
    {'http': '112.115.57.20:3128'},
    {'http': '121.41.171.223:3128'}
]
# proxies = [
#     'http://183.95.80.102:8080',
#     'http://123.160.31.71:8080',
#     'http://115.231.128.79:8080',
#     'http://166.111.77.32:80',
#     'http://43.240.138.31:8080',
#     'http://218.201.98.196:3128',
#     'http://112.115.57.20:3128',
#     'http://121.41.171.223:3128'
# ]

# 获取Cookie（Headers信息）
# 1）打开目标网站：https://www.dianping.com/，然后登录
# 2）浏览器的开发者工具=>检查=>网络/Network=>Doc（非Fetch/XHR）
# 3）点击左上角刷新按钮，‌在出现的请求中找到第一个接口并点击进入
# 4）在Headers中的Request Headers下可找到Cookie登录的信息
# 5）将Request Headers中的内容复制到代码中的Headers中（注意添加单引号）
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': 'fspop=test; cy=18; cye=shenyang; _lxsdk_cuid=190dfce5316c8-030c4213c41898-26031051-100200-190dfce5316c8; _lxsdk=190dfce5316c8-030c4213c41898-26031051-100200-190dfce5316c8; _hc.v=88f6077b-75ac-995c-93c9-6d1ff67da6ee.1721741759; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1721741761; HMACCOUNT=89BD5A42E2563851; s_ViewType=10; WEBDFPID=y3774uz547255419yuwux68y45z693z0809wuxvu35z979581811u64v-2037101801367-1721741801367ECGOKKEfd79fef3d01d5e9aadc18ccd4d0c95071008; ctu=896c85ee241e3ebf329cdf9404a98033f7ce2911cc00f75d255b646394007d74; _lx_utm=utm_source%3DBaidu%26utm_medium%3Dorganic; qruuid=3264f3b1-b24b-43da-9f3c-f8c10721fc37; dplet=94c8feb4210fb4398af689629a792552; dper=02029323fd366c430ca2451a798785415036a18e2dbfe4cfaaf1fa8db711d6a6001570fcaf700fe57b78ea1ac8c3ab5f424cea52d7811a6a0dda00000000a8210000d8cb6982c26a1f7c85bb9757bba96bc9287d718e38af69f3ff4f1eef1a30ba8d143646bd9157ee8cd156c45e994ddb15; ll=7fd06e815b796be3df069dec7836c3df; ua=uo; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1721747077; _lxsdk_s=190dfce5317-4ff-9d4-bf0%7C%7C877',
    'Host': 'www.dianping.com',
    'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}

# 随机代理IP
proxy = random.choice(proxies)
# 请求URL
# url = f'https://www.dianping.com/shenyang/ch10/p1'

# 网页分析：
# 第一页：https://www.dianping.com/shenyang/ch10/p1
# 第二页：https://www.dianping.com/shenyang/ch10/p2
# ......
# 爬取前10页
urls = [f'https://www.dianping.com/shenyang/ch10/p{str(i+1)}' for i in range(2)]

# 请求
def get_html_str(url: str):
    try:
        resp = requests.get(url=url, headers=headers, proxies=proxy)
        resp.raise_for_status()
        # 指定响应内容中的编码格式
        resp.encoding = resp.apparent_encoding
        html_str = resp.text
        # print(html_str)
        return html_str
    except Exception as e:
        print(e)


# 解析
def bs4_html_parser(html_str: str):
    # 保存结果
    data = []
    # 使用BeautifulSoup
    soup = BeautifulSoup(html_str, 'lxml')
    # 格式化HTML
    # print(soup.prettify())

    # 找到包含所有li标签的div
    div = soup.find(attrs={"id": "shop-all-list"})
    # 遍历所有li标签
    for li_tag in div.select('ul li'):
        # 详情链接
        href = li_tag.find(attrs={"class": "pic"}).a["href"]
        # 店名
        title = li_tag.find(attrs={"class": "tit"}).h4.text.strip()
        # 评价
        review_num = li_tag.find(attrs={"class": "review-num"}).b.text.strip()
        # 人均
        mean_price = li_tag.find(attrs={"class": "mean-price"}).b.text.strip()
        # 店铺分类
        shop_cate = li_tag.find(attrs={"data-click-name": "shop_tag_cate_click"}).span.text.strip()
        # 店铺地区
        shop_region = li_tag.find(attrs={"data-click-name": "shop_tag_region_click"}).span.text.strip()
        # 推荐菜
        recommends = [tag.text for tag in li_tag.find_all(attrs={"class": "recommend-click"})]
        # 团购
        if li_tag.find_all(attrs={"data-click-name": "shop_info_groupdeal_click"}):
            shop_groupdeal = [tag["title"].split('：', maxsplit=1)[1] for tag in li_tag.find_all(attrs={"data-click-name": "shop_info_groupdeal_click"})]
        else:
            shop_groupdeal = None
        # 优惠券
        if li_tag.find(attrs={"class": "tuan privilege"}):
            shop_promo = li_tag.find(attrs={"class": "tuan privilege"}).text.strip().split('：', maxsplit=1)[1]
        else:
            shop_promo = None
        # 输出
        data.append({"店名": title, "详情链接": href, "评价": review_num, "人均": mean_price, "店铺分类": shop_cate, "店铺地区": shop_region, "推荐菜": recommends, "团购": shop_groupdeal, "优惠券": shop_promo})
    return pd.DataFrame(data)


# 获取店铺详情
def get_details_info(url: str):
    # 抓取解析每个店铺的详情页
    soup = BeautifulSoup(get_html_str(url), 'html.parser')
    # 格式化HTML
    # print(soup.prettify())

    # 找到店铺详情div模块
    div = soup.find(attrs={"id": "basic-info"})
    # 店铺信息（包括在H1标签中）
    h1_info = re.split("\\s+", div.h1.text.strip())
    # 店名
    shop_name = h1_info[0].strip()
    # 分店信息
    branch_shop = h1_info[-1].strip()
    # 评价
    review_count = div.find(attrs={"id": "reviewCount"}).text.strip().split()[0]
    # 人均
    avg_price = div.find(attrs={"id": "avgPriceTitle"}).text.split(":")[1].strip().replace(' ', '')
    # 各项评分（动态加载）
    com_score = div.find(attrs={"id": "comment_score"}).text.strip().split("  ")
    # 星级/综合评分（动态加载）
    # score = div.find(attrs={"class": "mid-score score-40"}).text.strip()
    # 地址
    address = div.find(attrs={"id": "address"}).text.strip()
    # 电话
    tel = div.find(attrs={"class": "expand-info tel"}).text.split("：", maxsplit=1)[1].strip().split()
    # 营业时间
    business_hours = div.find(attrs={"class": "info info-indent"}).find(attrs={"class": "item"}).text.strip()
    # 输出
    return [{"店名": shop_name, "分店信息": branch_shop, "评价": review_count, "人均": avg_price, "各项评分": com_score, "综合评分": None, "地址": address, "电话": tel, "营业时间": business_hours}]



# url = "https://www.dianping.com/shop/H8lgCIV1v61ophNd"
# print(get_details_info(url))


if __name__ == '__main__':

    # 数据页
    data_res = pd.DataFrame()
    for url in urls:
        html_str = get_html_str(url)
        df = bs4_html_parser(html_str)
        data_res = pd.concat([data_res, df], ignore_index=True)
        time.sleep(1)
    print(data_res.to_string())
    # 详情页（根据数据页提供的详情链接获取）
    details = []
    for url in data_res['详情链接'].tolist():
        details += get_details_info(url)
    info_res = pd.DataFrame(details)
    print(info_res.to_string())





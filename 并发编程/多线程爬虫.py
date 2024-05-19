
import numpy as np
import pandas as pd
import requests
import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED, FIRST_COMPLETED

# 深入理解Python线程池ThreadPoolExecutor

# 从Python3.2开始，标准库为我们提供了concurrent.futures模块，它提供了ThreadPoolExecutor和ProcessPoolExecutor两个类，实现了对threading和multiprocessing的进一步抽象，不仅可以帮我们自动调度线程，还可以做到：
# 1）主线程可以获取某一个线程（或任务）的状态及返回值
# 2）当一个线程完成的时候，主线程能够立即知道
# 3）让多线程和多进程的编码接口一致
# 可见，concurrent.futures模块帮我们实现了并发的可见性和原子性

# 线程池的创建
# ThreadPoolExecutor提供了两种创建线程池的方法submit()和map()，map()底层实际上调用了submit()
# 1、submit()
# submit()用于提交要用给定参数执行的可调用对象，并返回表示可调用对象执行的Future实例
# submit()不是阻塞的，而是立即返回。通过submit()返回的Future实例可以通过done()方法判断任务是否结束
'''
submit(func, args)
func：任务函数对象  args：函数参数，元组类型
'''

# 线程池方法

# 1）cancel()方法可用于取消提交的排队等候的任务，如果任务已经在线程池中运行了，则无法取消
# 2）result()方法可用来获取任务的返回值。查看内部代码，发现这个方法是阻塞的

# 任务函数
def crawl(page):
    print(f"爬取{page+1}页已完成！")
    return page+1

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(crawl, page) for page in range(5)]
    # 判断任务是否完成
    dones = [future.done() for future in futures]
    print(dones)
    # 获取任务的返回值
    results = [future.result() for future in futures]
    print(results)

'''
爬取1页已完成！
爬取2页已完成！
[True, True, False, False, False]
爬取3页已完成！
爬取4页已完成！
爬取5页已完成！
[1, 2, 3, 4, 5]
'''
# 3）as_completed()方法可用于一次判断所有任务执行结果，as_completed()方法是一个生成器，在没有任务完成的时候，会阻塞，在有某个任务完成的时候，会yield这个任务，先完成的任务会先通知主线程

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(crawl, page) for page in range(5)]
    # 一次判断所有任务是否完成
    all_done = [future.result() for future in as_completed(futures)]
    print(all_done)

'''
爬取1页已完成！
爬取2页已完成！
爬取3页已完成！
爬取4页已完成！
爬取5页已完成！
[3, 2, 1, 5, 4]
'''

# 4）wait()方法可以让主线程阻塞，直到满足设定的条件。wait()方法接收3个参数，等待的任务序列、超时时间及等待条件
# 等待条件return_when默认为ALL_COMPLETED，表明要等待所有的任务都结束
# 等待条件还可以设置为FIRST_COMPLETED，表示第一个任务完成就停止等待

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(crawl, page) for page in range(5)]
    # 等待所有任务都完成
    wait(futures, return_when=ALL_COMPLETED, timeout=2)
    # 主线程
    print(threading.current_thread().name)

'''
爬取1页已完成！
爬取2页已完成！
爬取3页已完成！
爬取4页已完成！
爬取5页已完成！
MainThread
'''

# 2、map()
# map()函数会为可迭代对象的每个元素启动一个线程，以并发方式来执行任务函数，map()直接返回任务执行的可迭代结果
'''
map(func,iters)
func：任务函数对象  iters：需要操作的对象集合
'''
with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(crawl, [page for page in range(5)])
    results = [res for res in results]
    print(results)

'''
爬取1页已完成！
爬取2页已完成！
爬取3页已完成！
爬取4页已完成！
爬取5页已完成！
[1, 2, 3, 4, 5]
'''
# ThreadPoolExecutor让线程的使用更加方便，减少了线程创建/销毁的资源损耗，我们无需考虑线程间的复杂问题，方便主线程与子线程的交互

# 案例：多线程爬虫

url = 'https://example.com/api/'
referer = 'https://example.com'
headers = {'User-Agent': 'Mozilla/5.0', 'Referer': referer}
pages = 20

session = requests.session()

# 发起请求，返回数据
def get_json_data(page: int):
    params = {'page': page}
    try:
        resp = session.get(url=url, headers=headers, params=params)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        json_str = resp.text
        # 返回json格式的数据
        return json.loads(json_str)['rows']
    except:
        print(f"Page {page+1}: 爬取异常！")

# 数据处理（单线程）
def data_processor():
    df_res = pd.DataFrame()
    for page in range(pages):
        data = get_json_data(page)
        df = pd.DataFrame(data)
        print(f"Page: {page+1}/{pages}  数据条数: {len(df)}")
        df_res = pd.concat([df_res, df], ignore_index=True)
        time.sleep(0.1)
    print(f"数据总条数: {len(df_res)}")
    session.close()
    return df_res

# 数据处理
def data_processor(page):
    data = get_json_data(page)
    df = pd.DataFrame(data)
    print(f"Page: {page+1}/{pages}  数据条数: {len(df)}")
    return df

# 多线程爬取及数据处理
def mul_data_processor(max_size=os.cpu_count()):
    start = time.perf_counter()
    df_res = pd.DataFrame()
    with ThreadPoolExecutor(max_size) as executor:
        results = executor.map(data_processor, [i for i in range(pages)])
        for result in results:
            df_res = pd.concat([df_res, result], ignore_index=True)
    print(f"数据总条数: {len(df_res)}")
    print(f"爬取总用时: {round(time.perf_counter() - start, 2)}s")
    session.close()
    return df_res


if __name__ == '__main__':
    # 单线程爬取
    # data = data_processor()
    # 多线程爬取
    data = mul_data_processor(10)
    print(data.head(10).to_string())


# ThreadPoolExecutor默认的最大线程数为：
# max_workers = min(32, (os.cpu_count() or 1) + 4)
# 即CPU内核数量加4

# ThreadPoolExecutor默认的连接池大小为10，如果设置最大线程数max_workers>10，程序将发出警告：
# WARNING 2023-11-01 15:59:30,308 Connection pool is full, discarding connection: vginsights.com. Connection pool size: 10
# max_workers>10与max_workers=10的执行效率没有区别



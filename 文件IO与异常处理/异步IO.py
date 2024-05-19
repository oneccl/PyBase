"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/14
Time: 22:41
Description:
"""

# 异步IO（异步编程）

# 当发起一个IO操作时，并不需要等待它结束，程序可以去做其他事情，当这个IO操作结束时，会发起一个通知
# Python中可以使用asyncio模块进行异步编程，用于协程、网络爬虫、同步等

import asyncio

# 1、基本概念
# （1）event_loop 事件循环
# 事件循环是asyncio应用的核心，管理所有的事件
'''
1）创建新的事件循环
asyncio.new_event_loop()
2）获取当前线程中正在执行的事件循环
asyncio.get_running_loop()
3）并发执行任务
asyncio.gather()
4）向指定的事件添加一个任务
asyncio.run_coroutine_threadsafe()
5）返回没有执行的事件
asyncio.all_tasks()
'''

# （2）Future对象：一个Future代表一个异步运算的结果，线程不安全

# （3）Task对象：在运行某个任务的同时可以并发的运行其他任务
'''
1）创建：asyncio.create_task()
2）取消：cancel()
3）Task对象是否完成：done()
4）返回结果：result()
 - Task对象被完成，则返回结果
 - Task对象被取消，则引发CancelledError异常
 - Task对象的结果不可用，则引发InvalidStateError异常
5）添加回调，任务完成时触发：add_done_callback(task)
6）返回所有任务列表：asyncio.all_tasks()
7）返回当前任务：asyncio.current_task()
'''

# 2、异步IO的应用案例
# 1）Task任务回调
async def do_work():
    print("正在执行任务...")
    # 模拟阻塞1秒
    await asyncio.sleep(1)
    return "Data"

# 任务完成后的回调函数
def callback(task):
    # 处理任务返回的结果
    print(task.result())

# 创建一个事件event_loop
loop = asyncio.get_event_loop()

# 创建一个task
task = loop.create_task(do_work())
task.add_done_callback(callback)

# 将task加入到event_loop中
loop.run_until_complete(task)

# 2）并发任务
import time

async def do_work(t):
    await asyncio.sleep(t)
    return "暂停: " + str(t) + "s"

# 任务完成后的回调函数
def callback(future):
    # 处理返回的结果
    print(future.result())

# 创建一个事件event_loop
loop = asyncio.get_event_loop()

tasks = []
for i in range(5):
    task = loop.create_task(do_work(i))
    task.add_done_callback(callback)
    tasks.append(task)

# 计时
now = lambda: time.time()
start = now()
# 将task加入到event_loop中
loop.run_until_complete(asyncio.wait(tasks))
end = now()
print(f"用时: {end - start}")

# 3）同一回调并发任务
import functools

async def do_work(t):
    await asyncio.sleep(t)
    return "暂停: " + str(t) + "s"

# 多个Task任务完成后的回调
def callback(loop, gatheringFuture):
    print(gatheringFuture.result())
    # 终止事件循环
    loop.stop()

loop = asyncio.get_event_loop()

gather = asyncio.gather(do_work(1), do_work(2), do_work(3))
gather.add_done_callback(functools.partial(callback, loop))

# loop.run_forever()和loop.run_until_complete()
# run_until_complete()函数在Task任务执行完成后事件循环被停止
# run_forever()函数在Task任务执行完成后事件循环并不会被终止，在回调函数中使用loop.stop()函数可以将事件循环终止
loop.run_forever()


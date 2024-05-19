
# 1、基本概念

# 1）并行与并发
"""
并行：同一时间片并列执行
并发：不同时间片交替执行
"""

# 2）进程与线程
"""
a、进程是操作系统资源分配的基本单位，线程是CPU调度的基本单位
b、进程之间不共享全局变量，线程之间共享全局变量（资源竞争问题）
"""

# 3）并发问题解决思路
"""
1、构建队列、缓冲区：挨个执行不要拥挤
2、构建锁：哪个线程抢到独占执行，其它线程等待，该线程执行完成后释放锁
3、预处理：提前加载
4、并行：多核CPU
"""

# 2、Python中创建线程的3种方式：
"""
1）函数：使用threading模块创建：threading.Thread(func线程函数对象, args传递给线程函数的参数元组类型)
2）继承类：使用Threading模块创建，继承threading.Thread，重写run方法
3）线程池：使用concurrent.futures模块创建线程池
"""
import threading
import time

# 1）函数创建

# 线程执行的函数
def worker(thread_name, delay):
    count = 10
    while count > 0:
        time.sleep(delay)
        count -= 1
        print(f"{thread_name} 抢到了，剩余: {count}")

# 创建2个线程，并开启
thread1 = threading.Thread(target=worker, args=('Thread-1', 1))
thread2 = threading.Thread(target=worker, args=('Thread-2', 2))
thread1.start()
thread2.start()

# 多线程的使用：
'''
1、定义函数func
2、创建一个线程：t=threading.Thread(func, args)
3、启动线程：t.start()
4、线程阻塞、等待结束：t.join()
'''

# 2）继承类创建

# Lock对象
lock = threading.Lock()
# 共享资源
count = 10

class MyThread(threading.Thread):

    def __init__(self, thread_name, delay):
        super().__init__()
        self.thread_name = thread_name
        self.delay = delay

    def run(self) -> None:
        global count
        # 获取锁
        lock.acquire()
        while count > 0:
            time.sleep(self.delay)
            count -= 1
            print(f"{self.thread_name} 抢到了，剩余: {count}")
        # 释放锁
        lock.release()
        print("抢光了！")

# 创建2个线程，并开启
MyThread('Thread-1', 1).start()
MyThread('Thread-2', 2).start()

# Thread类提供的方法：
'''
thread.start() 启动线程
thread.join() 资源独占，阻塞线程，等待结束（thread占用所有资源，该线程运行完毕后，其他线程才能执行）
thread.getName() 返回线程名
thread.setName() 设置线程名
thread.daemon=True/thread.setDaemon(True) 设置thread线程为主线程的守护线程（会随着主线程的退出而退出，不论该线程是否执行完成）
'''

# 3）线程池创建
from concurrent.futures import ThreadPoolExecutor, wait

# 方式1：submit(func, args)
'''
submit(func, args)函数用于提交线程需要执行的任务（函数名和参数）到线程池中，并返回一个future对象
'''
with ThreadPoolExecutor(max_workers=2) as executor:
    # 多参数传递
    futures = [executor.submit(lambda args: worker(*args), (f"Thread-{i}", i+1)) for i in range(2)]
    # 等待所有任务都结束
    wait(futures)

# 方式2：map(func, iters)
'''
map(func,iters)函数会为iters的每个元素启动一个线程，以并发方式来执行func函数；使用map()函数还会自动获取返回值
'''
with ThreadPoolExecutor(max_workers=2) as executor:
    # 多参数传递
    results = executor.map(lambda args: worker(*args), [(f"Thread-{i}", i+1) for i in range(2)])
    # for result in results:
    #     print(result)

# 线程池方法：
'''
result() 获取任务执行的结果
cancel() 取消线程任务
'''

# 3、线程安全、线程同步（线程同步锁）
'''
某个线程获得锁，会阻塞该线程之后所有尝试获得该锁对象的线程，直到它被重新释放
通过加锁来确保多个线程在操作同一全局变量读写时的数据安全
'''
# 加锁：详见 2）继承类创建 中使用：
'''
# 创建Lock对象
lock = threading.Lock()
# 获取锁
lock.acquire()
# 释放锁
lock.release()
'''
# 创建线程
t1 = MyThread('Thread-1', 1)
t2 = MyThread('Thread-2', 2)
# 开启线程
t1.start()
t2.start()
# 等待所有线程执行完成
t1.join()
t2.join()

# 4、线程通信（queue同步队列）
'''
Python的queue模块可以实现线程通信；queue模块实现了多生产、消费者队列，特别适用于在多线程间安全的进行信息交换
该模块提供了多种可以利用的队列容器，如Queue（先进先出队列）、LifoQueue（先进后出队列）、PriortyQueue（优先级队列）
'''
# Queue（先进先出队列）常用方法
'''
# 创建一个容量为n的FIFO队列，若maxsize≤0，则队列无限大
queue.Queue(maxsize=n)
# 判断队列是否为空
queue.Queue.empty()
# 判断队列是否已满
queue.Queue.full()
# 添加元素到队列，block=True表示如果队列已满将阻塞线程，timeout用来设置线程阻塞的时长（秒），block=False且队列已满时，会报Full异常
queue.Queue.put(item, block=True, timeout=None)
# 从队列中取出一个元素，block=False且队列为空时，会报Empty异常
queue.Queue.get(block=True, timeout=False)
# 每个线程使用get方法从队列中取出一个元素，该线程通过调用task_done()表示该元素已处理完成
queue.Queue.task_done()
# 阻塞直到队列中所有元素都被处理完成，即队列中所有元素都已消费
queue.Queue.join()
'''
import queue
import random
from threading import Thread

# 创建一个容量为5的队列
q = queue.Queue(maxsize=5)
# 菜单
ls = [f"菜品{n}" for n in range(10)]

def cooking(chef_name: str):
    for i in range(3):
        # 从菜单中随机选取一个放入队列
        deal = random.choice(ls)
        q.put(deal, block=True)
        print(f"厨师 {chef_name} 生产 {deal} \t")

def eating(cust_name: str):
    for i in range(2):
        deal = q.get(block=True)
        print(f"顾客 {cust_name} 消费 {deal} \t")
        # 处理完成
        q.task_done()

# 创建并启动厨师A、B、C线程，创建并启动顾客1、2、3、4线程
chefs = [Thread(target=cooking, args=chef).start() for chef in ["A", "B", "C"]]
custs = [Thread(target=eating, args=str(cust)).start() for cust in range(1, 4)]
# 队列阻塞，直到所有线程对每个元素都调用了task_done
q.join()


# 定时任务、周期任务

# 初识
# 1、threading.Timer模块: 定时任务
"""
Timer(interval, func, args)
interval: 延迟多长时间执行任务(单位：s)
function: 要执行任务的函数对象
args/kwargs: 函数的参数(tuple)
"""
from datetime import datetime
from threading import Timer

# 等待3秒打印当前时间
def task():
    Timer(3, run, ()).start()

def run():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

task()

# 2、sched库
# sched是Python的标准库之一，sched是事件调度器，它通过Scheduler类来调度事件，可用于定时任务或周期任务

# 使用步骤：
"""
1）创建任务调度器
sched.scheduler(timefunc, delayfunc)
 - timefunc：代表当前时间
 - delayfunc：用于暂停运行的时间单元
一般使用默认参数：time.time和time.sleep
2）添加调度任务
scheduler提供了两个添加调度任务的函数：
①、enter(delay, priority, action, argument=(), kwargs={})：延迟一定时间执行任务
 - delay：延迟多长时间执行任务(单位：s)
 - priority：优先级，越小优先级越大
 - action：要执行任务的函数对象
 - argument、kwargs：函数的位置和关键字参数
②、enterabs(time, priority, action, argument=(), kwargs={})：指定时间执行任务
 - 任务会在time这时刻执行（time是绝对时间），其他参数与enter()相同
3）开启任务
scheduler.run()
"""

import sched
import time

# 2.1、周期任务

# 1）创建定时任务
scheduler = sched.scheduler(time.time, time.sleep)
# 需要执行的任务
def task():
    print("Hello, world!")

def run_task():
    # 2）指定需要执行的任务周期执行（每隔3秒）
    scheduler.enter(3, 1, task, ())
    scheduler.enter(3, 1, run_task, ())

# 3）开启任务
run_task()
scheduler.run()


# 2.2、定时任务
import smtplib
from email.mime.text import MIMEText

# 需要执行的任务
def send_email(subject, message, from_addr, to_addr, smtp_server):
    # 邮件的主体信息
    email = MIMEText(message)
    email['Subject'] = subject
    email['From'] = from_addr
    email['To'] = to_addr

    # 发邮件
    with smtplib.SMTP(smtp_server) as server:
        server.send_message(email)

def send_scheduled_email(subject, message, from_addr, to_addr, smtp_server, scheduled_time):
    # 1）创建定时任务
    scheduler = sched.scheduler(time.time, time.sleep)
    # 2）指定需要执行的任务
    scheduler.enterabs(scheduled_time, 1, send_email, argument=(subject, message, from_addr, to_addr, smtp_server))
    # 3）开启任务
    scheduler.run()

subject = 'Test Email'
message = 'This is a test email'
from_addr = 'test@example.com'
to_addr = 'test@example.com'
smtp_server = 'smtp.test.com'

# 一分钟之后执行任务
scheduled_time = time.time() + 60
send_scheduled_email(subject, message, from_addr, to_addr, smtp_server, scheduled_time)



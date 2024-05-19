
# 4、APScheduler库：定时任务、周期任务
# APScheduler（Advanced Python Scheduler）是一个基于Quartz的轻量级的Python定时任务调度框架
# APScheduler支持三种调度任务：固定时间间隔，固定时间点（日期），Linux下Crontab命令；同时还支持异步执行、后台执行调度任务

from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 使用三部曲（周期任务）
# 1）创建一个调度器：创建后台执行的schedulers
scheduler = BlockingScheduler()
# 2）添加一个调度任务，调度方法为timedTask，触发器选择interval(间隔性)，时间间隔为3秒
scheduler.add_job(task, 'interval', seconds=3)
# 3）启动调度任务
scheduler.start()

# APScheduler基础组件
# APScheduler的四大组件：调度器(scheduler)，作业存储(job store)，触发器(trigger)，执行器(executor)
'''
1）调度器(scheduler)：整个调度的总指挥官；调度器会合理安排作业存储器、执行器、触发器进行工作，并进行添加、修改和删除任务等
2）作业存储(job store)：任务持久化仓库，默认保存在内存中，也可保存在各种数据库中（任务中的数据序列化及反序列化）
3）触发器(trigger)：描述调度任务被触发的条件，是完全无状态的
4）执行器(executor)：负责处理作业的运行，提交指定的可调用对象到线程池或进程池运行；当作业完成时，执行器将通知调度器
'''
# 4.1、调度器(scheduler)
# APScheduler提供了7种调度器，能够满足各种场景的需要。如后台执行某个操作，异步执行操作等
'''
BlockingScheduler: 调度器在当前进程的主线程中运行，会阻塞当前线程，不能立即返回
BackgroundScheduler: 调度器在后台线程中运行，不会阻塞当前线程
AsyncIOScheduler: 结合asyncio模块（异步框架）一起使用
GeventScheduler: 使用gevent（高性能的Python并发框架）作为IO模型和GeventExecutor配合使用
TornadoScheduler: 使用Tornado（Web框架）的IO模型，用ioloop.add_timeout完成定时唤醒
TwistedScheduler: 结合TwistedExecutor，用reactor.callLater完成定时唤醒
QtScheduler: 用于构建Qt应用程序，需使用QTimer完成定时唤醒
'''
# 4.2、触发器(trigger)
# APScheduler有三种内建trigger：date触发器、interval触发器和cron触发器
# add_job()基本参数：
'''
add_job(func, trigger, id, max_instances, args, kwargs)
 - func：要执行任务的函数对象
 - trigger：触发器类型
 - id：Job的唯一标识
 - max_instances：并发执行最多任务数
 - args：函数的参数(tuple)
'''
# 1）date触发器：定时任务
# date触发器是最基本的一种调度，作业任务只会执行一次；它表示在特定的时间点触发
# add_job()参数：
'''
- run_date: 作业的运行日期或时间，datetime或str类型
- timezone: 指定时区，datetime.tzinfo或str类型
'''
from datetime import datetime
from datetime import date
from apscheduler.schedulers.blocking import BlockingScheduler

def task(dt):
    print(dt)

scheduler = BlockingScheduler()

# 在2023-08-24时刻执行一次任务
scheduler.add_job(task, 'date', run_date=date(2023, 8, 24), args=(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
# 在2023-08-24 14:00:00时刻执行一次任务
scheduler.add_job(task, 'date', run_date=datetime(2023, 8, 24, 14, 0, 0), args=(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
# 在2023-08-24 11:00:01时刻执行一次任务
scheduler.add_job(func=task, trigger='date', run_date='2023-08-24 11:00:01', args=(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

scheduler.start()

# 2）interval触发器：周期任务
# interval触发器用于固定时间间隔触发
# add_job()参数：
'''
- weeks：间隔几周，int类型
- days：间隔几天，int类型
- hours：间隔几小时，int类型
- minutes：间隔几分钟，int类型
- seconds：间隔多少秒，int类型
- start_date、end_date：开始日期(包含)、结束日期(包含)，datetime或str类型
- timezone：时区，datetime.tzinfo或str类型
'''
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

scheduler = BlockingScheduler()
# 每隔2s执行一次任务
scheduler.add_job(task, 'interval', seconds=2)
# 从2023-08-24 14:00:00开始执行，到2024-12-31 14:00:01结束, 每隔1天执行一次任务
scheduler.add_job(task, 'interval', days=1, start_date='2023-08-24 14:00:00', end_date='2024-12-31 14:00:01')

scheduler.start()

# 3）cron触发器
# cron触发器用于在特定时间周期性地触发，与Linux crontab格式兼容；它是功能最强大的触发器
# add_job()参数：
'''
- year：年，4位数字，int或str类型
- month：月(范围1-12)，int或str类型
- day：日(范围1-31)，int或str类型
- week：周(范围1-53)，int或str类型
- day_of_week：周内第几天或者星期几(范围0-6或mon,tue,wed,thu,fri,sat,sun)，int或str类型
- hour：时(范围0-23)，int或str类型
- minute：分(范围0-59)，int或str类型
- second：秒(范围0-59)，int或str类型
- start_date、end_date：开始日期(包含)、结束日期(包含)，datetime或str类型
- timezone：时区，datetime.tzinfo或str类型
'''
# cron触发器的参数支持算数表达式
'''
表达式        字段          描述
*            任何          每个值都触发
*/a          任何          每隔a触发一次
a-b          任何          在[a,b]区间任何一个时间触发
a-b/c        任何          在[a,b]区间每隔c触发一次
x,y,z        任何          多个表达式组合
xth y        day           第x星期y触发
last x       day           最后一个星期x触发
last         day           在一个月的最后一天触发
'''
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

scheduler = BlockingScheduler()

# 在每年的1-3、7-9月份中的每个星期一、二中的00:00、01:00、02:00和03:00执行任务
scheduler.add_job(task, 'cron', month='1-3,7-9', day='0, tue', hour='0-3')
# 从星期一到星期五的10:30执行任务，直到2024-12-31 00:00:00
scheduler.add_job(task, 'cron', day_of_week='mon-fri', hour=10, minute=30, end_date='2024-12-31')

scheduler.start()

# 4.3、作业存储(job store)
# APScheduler任务存储器有两种：内存（默认）、数据库
'''
MemoryJobStore：没有序列化，任务存储在内存中，增删改查都是在内存中完成
SQLAlchemyJobStore：使用SQLAlchemy-ORM框架作为存储方式
MongoDBJobStore：使用Mongodb作为存储
RedisJobStore：使用Redis作为存储
'''
# 1）添加Job
# 方式1：add_job()：该方法返回一个apscheduler.job.Job的实例，可以用来改变或移除Job
# 方式2：scheduled_job()装饰器：只适用于应用运行期间不会改变的Job
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

@scheduler.scheduled_job(task, 'interval', seconds=3)
def task():
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

scheduler = BlockingScheduler()
scheduler.start()

# 2）移除Job
# 方式1：remove_job()：根据Job的id移除（需要在add_job()时指定一个id）
# 方式2：job.remove()：直接移除

# 3）获取Job列表
# get_jobs()：该方法用于获取当前调度器中的所有Job列表

# 4）修改Job
# Job.modify()、modify_job(id)：用于修改Job属性（id不能被修改）
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

scheduler = BlockingScheduler()

job = scheduler.add_job(task, 'interval', seconds=2, id='job-1')
scheduler.start()

# 将触发时间间隔修改成5s
# 方式1：
scheduler.modify_job('job-1', seconds=5)
# 方式2：
job.modify(seconds=5)

# 5）关闭Job
# 默认情况下调度器会等待所有正在运行的作业完成后，关闭所有的调度器和作业存储；若不想等待，可以将wait设置为False
scheduler.shutdown()
scheduler.shutdown(wait=False)

# 4.4、执行器(executor)
# APScheduler常用的executor有两种：ProcessPoolExecutor和ThreadPoolExecutor
# 设置job store使用Mongodb存储和executor的使用示例：
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

def task():
    print("Job is Running.")

client = MongoClient('localhost', 27017)

# 作业存储
jobstores = {
    'mongo': MongoDBJobStore(collection='job', database='test', client=client),
    'default': MemoryJobStore()
}
# 执行器
executors = {
    'default': ThreadPoolExecutor(10),         # 10个线程
    'processpool': ProcessPoolExecutor(3)      # 3个进程
}
defaults = {
    'coalesce': False,       # 任务相同触发多次
    'max_instances': 3       # 每个任务最多同时触发3次
}

scheduler = BlockingScheduler(jobstores=jobstores, executors=executors, job_defaults=defaults)
scheduler.add_job(task, 'interval', seconds=5)

try:
    scheduler.start()
except SystemExit:
    client.close()

# 4.5、异常监听（Event事件）、异常处理
# Event是APScheduler在执行任务时触发的事件，用户可以自定义一些函数来监听这些事件，当触发某些Event时(任务抛出异常后)，做一些具体操作
# 常用Event：
# Job执行异常：EVENT_JOB_ERROR、Job执行时间错过：EVENT_JOB_MISSED、Job执行成功：EVENT_JOB_EXECUTED
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_EXECUTED
from datetime import datetime

scheduler = BlockingScheduler()

count = 5
@scheduler.scheduled_job('interval', seconds=2)
def task():
    global count
    count -= 1
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"\t 10 / {count} = {10 / count}")

def err_listener(Event):
    if not Event.exception:
        print("Job Running...")
    else:
        print("异常处理！")

scheduler.add_listener(callback=err_listener, mask=EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_EXECUTED)

try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown(wait=False)

# 4.6、并发执行
# 默认情况下，APScheduler会将多个任务串行执行（一个任务结束后才会执行下一个任务）
# 如果希望并发执行多个任务，可以设置使用max_instances参数
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def task1():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\tTask1 已执行！")

def task2():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\tTask2 已执行！")

scheduler = BlockingScheduler()

# 设置并发任务，最多并发执行3个任务
scheduler.add_job(task1, 'interval', seconds=2, id='job-1', max_instances=2)
scheduler.add_job(task2, 'interval', seconds=3, id='job-2', max_instances=2)

scheduler.start()


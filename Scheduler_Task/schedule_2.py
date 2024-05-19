
# 3、schedule库
# schedule是Python中的一个轻量级定时任务调度库，可以完成每隔时间、特定日期的周期任务和定时任务

import schedule
from datetime import datetime, timedelta

# 3.1、周期任务
# 方式1：
def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

schedule.every(3).seconds.do(task)                # 每3秒执行一次
schedule.every().day.at("00:00").do(task)         # 每天凌晨运行一次
schedule.every().monday.at("00:00").do(task)      # 每周一凌晨运行一次
schedule.every().wednesday.at("14:00").do(task)   # 每周三14:00运行一次
# 每小时运行一次，2024-10-31 11:59截止（停止）
schedule.every(1).hours.until("2024-10-31 11:59").do(task)
# 每小时运行一次，8个小时后停止
schedule.every(1).hours.until(timedelta(hours=8)).do(task)

while True:
    schedule.run_pending()


# 方式2：使用装饰器
@schedule.repeat(schedule.every(3).seconds)
def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

while True:
    schedule.run_pending()

# 取消任务
# schedule.CancelJob
# 取消所有任务
# schedule.clear()

# 3.2、定时任务
# 当天21:50执行一次
@schedule.repeat(schedule.every().day.at('21:50'))
def task():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return schedule.CancelJob

while True:
    schedule.run_pending()


# 3.3、异常处理
# schedule遇到异常会直接抛出，这会导致后续所有的任务被中断执行，因此我们需要捕捉到这些异常
# 可以手动捕捉，也可以使用装饰器解决 todo
import functools

def catch_exceptions(cancel_on_failure=False):
    def catch_exceptions_decorator(job_func):
        @functools.wraps(job_func)
        def wrapper(*args, **kwargs):
            try:
                return job_func(*args, **kwargs)
            except:
                import traceback
                print(traceback.format_exc())
                if cancel_on_failure:
                    return schedule.CancelJob
        return wrapper
    return catch_exceptions_decorator

count = 5
@catch_exceptions(cancel_on_failure=True)
def task():
    global count
    print(10 / count)
    count -= 1

schedule.every(2).seconds.do(task)

while True:
    schedule.run_pending()


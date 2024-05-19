

# dateutil模块

# dateutil模块是由Gustavo Niemeyer在2003年编写而成的对日期时间操作的第三方模块
# dateutil模块对Python内置的datetime模块进行扩展时区和解析
# dateutil库主要有两个模块：parser和rrule，其中parser可以将字符串解析成datetime，而rrule则是根据定义的规则来生成datetime

# dateutil模块特点
# - 能够计算日期时间相对增量，例如下周、下个月、明年、每月的最后一周等
# - 可以计算两个给定日期和/或日期时间对象之间的相对增量
# - 支持多种时区格式文件的解析，例如UTC时区、TZ时区等
# - 支持包括RFC字符串或其他任何字符串格式的通用日期时间解析

# 官方文档：http://labix.org/python-dateutil

# 安装：pip install python-dateutil

from dateutil import parser, rrule, relativedelta
# dateutil.parser：将字符串解析成datetime
# dateutil.rrule：将参数输出datetime.datetime格式的时间
# dateutil.relativedelta：日期时间偏移量

# 1）parser.parse(str)
# 将字符串解析成datetime，字符串可以很随意，可以用日期时间的英文单词，可以用横线、逗号、空格等做分隔符

# 没指定时间默认0点，没指定日期默认当天，没指定年份默认当年
# 当年份放在前面时，只能按年月日的顺序
print(parser.parse('2023-11-29'))          # 2023-11-29 00:00:00
print(parser.parse('10:45:52'))            # 2023-11-29 10:45:52
print(parser.parse('20231129'))            # 2023-11-29 00:00:00
print(parser.parse('2023.11.29'))          # 2023-11-29 00:00:00
print(parser.parse('2023/11/29'))          # 2023-11-29 00:00:00

# fuzzy：开启模糊匹配，过滤无法识别的时间日期字符
print(parser.parse('Today is 11-29 10:45, I feel good.', fuzzy=True))   # 2023-11-29 10:45:00

# 当只有月日时，parser会将分隔符前面的数字解析为月份，后面的为日
# 当有年份时，在前面的月份超出范围时，会自动判断哪个是月哪个是日
# 11.29解析结果异常，11-29、11/29可正常解析
print(parser.parse('11-29'))               # 2023-11-29 00:00:00
print(parser.parse('11/29/2023'))          # 2023-11-29 00:00:00

# 当前面的月份超过12时，parser会自动识别月和日
print(parser.parse('29/11/2023'))          # 2023-11-29 00:00:00

# 当分隔符为逗号时，只有月日时，要把月放在后面
# 当分隔符为逗号时，有年份时，年份要放在后面，要把月放在前面
print(parser.parse('29,11'))               # 2023-11-29 00:00:00
print(parser.parse('11,29,2023'))          # 2023-11-29 00:00:00

# 识别英文的月、日
print(parser.parse('November 29'))         # 2023-11-29 00:00:00
print(parser.parse('November 1st'))        # 2023-11-01 00:00:00
print(parser.parse('November 29 2023'))    # 2023-11-29 00:00:00
print(parser.parse('2023 November29'))     # 2023-11-29 00:00:00
print(parser.parse('11:45 AM'))            # 2023-11-29 11:45:00

# 2）rrule.rrule(freq,dtstart,interval,wkst,count,until,by)
'''
- freq：单位，可选的值为YEARLY、MONTHLY、WEEKLY、DAILY、HOURLY、MINUTELY、SECONDLY，即年月日周时分秒
- dtstart、until：开始和结束时间，时间格式datetime.datatime类型
- interval：间隔
- wkst：周开始时间
- count：生产时间的个数
- by：指定匹配的周期，例如，byweekday=(MO,TU)：只有周一周二的匹配，取值如下：
  - bysetpos：必须为整数或者整数序列，设置匹配的周期频率
  - bymonth：设置匹配的月份
  - bymonthday：设置匹配每月的日期
  - byyearday：设置匹配每年的天数
  - byweekno：设置匹配第几周
  - byweekday：MO,TU,WE,TH,FR,SA,SU
  - byhour：设置匹配小时
  - byminute：设置匹配分钟
  - bysecond：设置匹配秒数
'''

# 生成一个连续的日期列表
print(list(rrule.rrule(rrule.DAILY, dtstart=parser.parse('2023-11-29'), until=parser.parse('2023-12-3'))))
'''
[datetime.datetime(2023, 11, 29, 0, 0), datetime.datetime(2023, 11, 30, 0, 0), datetime.datetime(2023, 12, 1, 0, 0), datetime.datetime(2023, 12, 2, 0, 0), datetime.datetime(2023, 12, 3, 0, 0)]
'''

# 间隔一天
print(list(rrule.rrule(rrule.DAILY, interval=2, dtstart=parser.parse('2023-11-29'), until=parser.parse('2023-12-3'))))
'''
[datetime.datetime(2023, 11, 29, 0, 0), datetime.datetime(2023, 12, 1, 0, 0), datetime.datetime(2023, 12, 3, 0, 0)]
'''

# 只保留前3个元素
print(list(rrule.rrule(rrule.DAILY, count=3, dtstart=parser.parse('2023-11-29'), until=parser.parse('2023-12-3'))))
'''
[datetime.datetime(2023, 11, 29, 0, 0), datetime.datetime(2023, 11, 30, 0, 0), datetime.datetime(2023, 12, 1, 0, 0)]
'''

# 只取周六和周日日期时间
print(list(rrule.rrule(rrule.DAILY, byweekday=(rrule.SA, rrule.SU), dtstart=parser.parse('2023-11-29'), until=parser.parse('2023-12-3'))))
'''
[datetime.datetime(2023, 12, 2, 0, 0), datetime.datetime(2023, 12, 3, 0, 0)]
'''

# 以月为间隔，生成3个月
print(list(rrule.rrule(rrule.MONTHLY, count=3, dtstart=parser.parse('2023-11-01'))))
'''
[datetime.datetime(2023, 11, 1, 0, 0), datetime.datetime(2023, 12, 1, 0, 0), datetime.datetime(2024, 1, 1, 0, 0)]
'''

# rrule计算日期时间差
# rrule可计算出两个datetime对象间相差的年月日等时间数量
print(rrule.rrule(rrule.DAILY, dtstart=parser.parse('20231129'), until=parser.parse('20231203')).count())    # 5
# 不足N个月的，按N个月计算；不满整月的，按N-1个月计算
print(rrule.rrule(rrule.MONTHLY, dtstart=parser.parse('20231101'), until=parser.parse('20240115')).count())  # 3

# 3）relativedelta.relativedelta()
# relativedelta主要用于日期时间偏移

# datetime.timedelta与relativedelta.relativedelta()
from datetime import datetime, timedelta

# timedelta仅支持：weeks、days、hours、minutes、seconds，不支持月、年
# datetime.timedelta()计算上周
print(datetime.strftime(datetime.now() - timedelta(weeks=1), '%Y%m%d'))   # 20231122
# relativedelta.relativedelta()计算上周
print(datetime.strftime(datetime.now() - relativedelta.relativedelta(weeks=1), "%Y%m%d"))   # 20231122

# 计算上月初、计算上N月初
print(datetime.strftime(datetime.now() - relativedelta.relativedelta(months=1), "%Y%m01"))   # 20231001
# 计算上月末、计算上N月末
print(datetime.strftime(datetime.strptime(datetime.now().strftime('%Y-%m-01'), '%Y-%m-%d') - relativedelta.relativedelta(days=1), '%Y%m%d'))   # 20231031





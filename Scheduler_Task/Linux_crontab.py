
# crontab（Linux命令）

# crontab是一个用于设置周期性被执行的指令。其守护进程为crond；crontab分为两种配置模式，一种为用户级的crontab，一种为系统级的crontab
# crontab命令是Unix和Linux用于设置周期性被执行的指令，是互联网很常用的技术，很多任务都会设置在crontab循环执行
# 如果不使用crontab，那么任务就是常驻程序，这对你的程序要求比较高，一是要求你的程序是24x7小时不宕机，二是要求你的调度程序比较可靠
# 实际工作中，90%的程序都没有必要花这么多时间和精力去解决上面的两个问题，只需要写好自己的业务逻辑，通过crontab工业级程序去调度就行了，crontab的可靠性、健壮性与稳定性是毫无疑问的

# 1、用户级crontab
"""
用户使用新建周期性工作调度时，使用的crontab命令：crontab -e
此时会进入vi的编辑工作界面，每项工作都是一行，编辑完毕之后输入:wq保存退出即可
所有用户都可以使用，普通用户只能为自己设置计划任务，然后自动写入/var/spool/cron/username
"""
# 用户控制文件
'''
/etc/cron.allow：将可以使用crontab的用户写入，仅该文件内的用户可以使用crontab，相当于白名单 
/etc/cron.deny：将禁止使用crontab的用户写入，仅该文件内的用户禁止使用crontab，相当于黑名单
其中/etc/cron.allow优先级大于/etc/cron.deny，建议二者仅使用一个
'''
# 命令：
'''
crontab [-u username] [-l|-e|-r] 
参数： 
-u：只有root才能执行这个任务，即帮其他用户新建/删除crontab工作调度 
-e: 调用vi编辑crontab的工作内容 
-l: 列出crontab的工作内容
-r: 删除所有crontab的工作内容（若仅要移除一项，则使用-e编辑）
'''
# 语法：
'''
.---------------- 分钟 (0 - 59) 
| .-------------- 小时 (0 - 23)
| | .------------ 日期(天) (1 - 31)
| | | .---------- 月份 (1 - 12) OR jan,feb,mar,apr,...
| | | | .-------- 周几 (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
| | | | |
* * * * * 命令（编辑任务添加在文件后面）
解释：
*   任何时间
,   分段时间，时间1、时间2和时间n
-   一段时间范围
/n  每隔n时间
'''
# 即：分 时 日 月 周 任务命令

# 语法与系统级crontab相似，不同点在于此处不需要指定执行用户，而系统级crontab(/etc/crontab)中需要

# 例如：
'''
*/10 * * * * /home/test.sh          # 每隔10分钟以当前用户执行一次脚本
0 2 * * * /home/test.sh             # 每天2点以当前用户执行一次脚本
0 5,17 * * * /home/test.sh          # 每天5点、17点以当前用户执行一次脚本
0 17 * * sun /home/test.sh          # 每周日17点以当前用户执行一次脚本
0 4,17 * * sun,mon /home/test.sh    # 每周一、周日以当前用户执行一次脚本
0 2 1 4 * /home/test.sh             # 在4月1号凌晨2点0分以当前用户执行一次脚本
@reboot /home/test.sh               # 系统重启时以当前用户执行一次脚本
'''

# 2、系统级crontab
'''
系统级crontab一般用于系统的例行性任务，这种方法更加方便与直接直接给其他用户设置计划任务，而且还可以指定执行shell等。配置文件为/etc/crontab，该文件仅root用户能够编辑
'''
# 编辑/etc/crontab文件（语法）
'''
SHELL=/bin/bash                     # 这里是指定使用哪种shell接口 
PATH=/sbin:/bin:/usr/sbin:/usr/bin  # 这里指定文件查找路径 
MAILTO=root                         # 如果有额外的STDOUT，以Email将数据送给谁，可以指定系统用户，也可以指定Email地址
.---------------- 分钟 (0 - 59) 
| .-------------- 小时 (0 - 23)
| | .------------ 日期(天) (1 - 31)
| | | .---------- 月份 (1 - 12) OR jan,feb,mar,apr,...
| | | | .-------- 周几 (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
| | | | |
* * * * * 用户名 命令（编辑任务添加在文件后面）
'''
# 即：分 时 日 月 周 执行用户 任务命令
# 注意：crontab -e命令会检查语法，而vim编辑/etc/crontab则不会，crontab -e不需要写执行者用户名，而/etc/crontab需要指定
# 例如：
'''
*/10 * * * * root /home/test.sh    # 添加一个计划任务，每隔10分钟以root身份执行一次脚本
'''
# 注意：不要漏掉执行用户root(用户级的crontab中不需要指定)，否则会在/var/log/cron日志中出现"ERROR (getpwnam() failed)"错误，计划任务无法正常运行

# crontab命令的原理与重启服务
'''
当使用者使用crontab命令创建工作编排后，该项工作将会被记录到/var/spool/cron/中（以账号作为判别）例如blue使用crontab后，他的工作会被记录到/var/spool/cron/blue中，cron运行的每一项工作日志都保存在/var/log/cron中
crond服务的最低侦测限制是分钟，所以cron会每分钟去读取一次/etc/crontab与/var/spool/cron里面的数据内容，因此，只要编辑完/etc/crontab文件保存后，cron的配置就会自动的来执行
一般来说Linux下的crontab会自动帮我们每分钟重新读取一次/etc/crontab的例行工作事项，但是出于某些原因，由于crontab是读到内存当中的，所以修改完/etc/crontab之后，可能并不会马上执行，这时候需要重启crontab服务
'''
# 以CentOS 7为例：
'''
systemctl restart crond.service       # 重启服务 
systemctl start crond.service         # 启动服务 
systemctl stop crond.service          # 停止服务 
systemctl reload crond.service        # 重载配置 
systemctl status crond.service        # 服务状态
'''

# crontab命令后台执行
'''
当在前台运行某个作业时，终端被该作业占据；可以使用&命令将作业放到后台执行：
*/10 * * * * /home/test.sh &
'''
# 在后台运行作业时要注意：需要用户交互的命令不要放在后台执行，因为这样你的机器将会在那里等待
# 后台运行的作业一样会将结果输出到屏幕上，干扰你的工作。若产生大量输出，最好使用如下方法将其输出重定向到某个文件中：
'''
*/10 * * * * /home/test.sh >out.file 2>&1 &
'''
# 2>&1：将所有的标准输出和错误输出都重定向到out.file文件中

# crontab命令2>&1含义
# 语法格式：command > file 2>&1 或 command 1> file 2>&1
# 含义：首先是command > file将标准输出重定向到file中，2>&1是标准错误拷贝了标准输出，也就是同样被重定向到file中，最终结果就是标准输出和错误都被重定向到file中
# 例如：
'''
0 2 * * * /home/test.sh >/dev/null 2>&1 &
0 2 * * * /home/test.sh 2>/home/out.file 2>&1 &
'''
# 解释：将错误输出2重定向到标准输出1，然后将标准输出1全部保存到/dev/null文件中（清空，即抛弃）或保存到out.file文件中
'''数字解释
0: 键盘输入
1: 标准输出
2: 错误输出
'''

# 语法使用注意事项
'''
执行路径必须使用绝对路径，否则可能无法正常执行
周与日、月不能共存，即可以分别以周或日、月为单位进行循环，但不可指定"几月几号且为星期几"的模式工作
'''

# 案例：每晚23:00定时执行Python脚本，备份MySql数据库（注意不要忘了“分”位置上的“0”）
'''
0 23 * * * python /var/www/html/crontab_python/back_db.py >/dev/null 2>&1
'''


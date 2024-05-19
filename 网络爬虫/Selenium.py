
# Selenium（Web Browser Automation）

# Selenium的初衷是Web应用自动化测试。Selenium广泛应用于爬虫，爬虫需要让浏览器自动运行网址来获取我们需要的内容
# Selenium不是单个软件，它是由一系列的工具组成

# 1、Selenium环境搭建

# 1）下载浏览器驱动（WebDriver）：用于驱动浏览器运行
# Chrome浏览器的WebDriver（chromedriver.exe）下载安装配置：

# （1）查看Chrome浏览器版本
# 设置 -> 关于Chrome -> 版本
# （2）下载对应版本对应操作系统的驱动
# 淘宝镜像(114及之前版本)：http://npm.taobao.org/mirrors/chromedriver/
# 官方镜像(指定版本下载)：https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.159/win64/chromedriver-win64.zip
# 其它：https://googlechromelabs.github.io/chrome-for-testing/
# （3）安装目录
# 将解压的chromedriver.exe复制放在Chrome浏览器安装目录下的Application目录中或Python解释器的安装目录下（与python.exe同级）
# （4）配置系统环境变量Path：添加chromedriver.exe绝对路径，如果放在Python相关目录下该步骤可省略
# # Application目录（右键图标打开文件所在位置）
# C:\Users\cc\AppData\Local\Google\Chrome\Application
# # Python解释器所在目录（或Scripts下）
# E:\Program Files\Python\Python310
# E:\Program Files\Python\Python310\Scripts
# （5）验证：Win+R，打开cmd命令行，使用如下命令验证：chromedriver.exe

# 2）安装：pip install selenium

# 2、基本操作

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains, Keys  # 动作类
from selenium.webdriver.support.wait import WebDriverWait  # 等待类
from selenium.webdriver.support import expected_conditions as EC  # 等待条件类
from selenium.webdriver.support.select import Select  # Select类
import time

# # 1）页面等待加载
# # 强制等待10s
# time.sleep(10)
#
# # 显式等待（Explicit Wait）：等待特定条件达成
# # 最长等待时间10s，EC.presence_of_element_located()是一个预定义的条件，用于等待元素出现在页面中
# element = WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, 'expr'))
# )
#
# # 隐式等待（Implicit Wait）：等待一定时间段
# # 最长等待时间10s，如果元素未找到将继续等待直到最长时间
# driver.implicitly_wait(10)
#
# # 设置页面异步加载超时时间10s
# driver.set_page_load_timeout(10)
#
# # 2）调整浏览器窗口大小
# # 窗口最大化
# driver.maximize_window()
# # 窗口最小化
# driver.minimize_window()
# # 调整窗口到指定尺寸
# driver.set_window_size(width,height)
#
# # 3）获取页面内容
# # 页面标题
# element.title
# # 页面HTML源码
# element.page_source
# # 页面链接
# element.current_url
#
# # 4）文本输入、清除与提交
# # 文本输入
# element.send_keys('text')
# # 文本清空
# element.clear()
# # 文本提交
# # submit()只能用于包含属性type='submit'的标签，并且嵌套在form表单中。也可以使用click()代替submit()使用
# element.submit()
#
# # 5）页面前进、后退与刷新
# # 前进一页
# driver.forward()
# # 后退一页
# driver.back()
# # 页面刷新
# driver.refresh()
#
# # 6）窗口切换
# # 获取当前窗口的句柄（窗口唯一标识）
# driver.current_window_handle
# # 获取所有打开页面的句柄（列表）
# driver.window_handles
# # 切换到指定页面，xx代表页面句柄
# driver.switch_to.window('xx')
# # 切换到内联框架页面，xx代表内联框架标签的定位对象
# driver.switch_to.frame('xx')
# # 切回到内联框架的上一级，即从内联框架切出
# driver.swith_to.parent_frame()
# # 切换到页面弹窗
# driver.switch_to.alert
#
# # 7）获取标签元素的属性值
# # 获取标签属性值，xx为标签属性名
# element.get_attribute("xx")
# # 获取标签内文本
# element.text
# # 获取元素X、Y位置
# element.location.get('x')
# # 获取元素的宽、高大小
# element.size.get('width')
#
# # 8）下拉列表操作
# # 判断标签元素xx是否为下拉列表元素，是返回Select对象，不是报错
# select = Select('xx')
# # 通过下拉列表value属性的值选择选项
# select.select_by_value("xx")
# # 通过下拉列表文本内容选择选项
# select.select_by_visible_text("xx")
# # 通过下拉列表索引号N选则选项，从0开始
# select.select_by_index(N) 或
# select.options[N].click()
# # 下拉列表内所有option标签
# select.options
#
# # 9）弹窗操作
# # 获取弹窗对象
# alert = driver.switch_to.alert
# # 弹窗内容
# alert.text
# # 接受弹窗
# alert.accept()
# # 取消弹窗
# alert.dismiss()
#
# # 10）鼠标操作
# # 鼠标悬停，x代表定位到的标签
# actionChains.move_to_element('x')
# # 单击
# element.click()
# actionChains.click('x')
# # 双击
# element.double_click()
# actionChains.double_click('x')
# # 右击
# element.context_click()
# actionChains.context_click('x')
# # 执行所有存储在ActionChains()类中的行为，做最终的提交
# perform()
# # 元素拖放，将源元素拖动到目标元素的位置
# actionChains.drag_and_drop(source_element, target_element).perform()
# # 按下鼠标左键(按住元素)
# actionChains.click_and_hold(element).perform()
# # 指定offset移动元素，如xoffset随变量var移动，yoffset=0表示Y轴不移动
# actionChains.move_by_offset(xoffset=var, yoffset=0).perform()
# # 松开鼠标
# actionChains.release().perform()
#
# # 11）键盘操作
# # 执行回退键Backspace
# element.send_keys(Keys.BACK_SPACE)
# # 全选`在这里插入代码片`
# element.send_keys(Keys.CONTROL,'a')
# # 剪切
# element.send_keys(Keys.CONTROL,'x')
# # 复制
# element.send_keys(Keys.CONTROL,'c')
# # 粘贴
# element.send_keys(Keys.CONTROL,'v')
#
# # 12）执行JS代码
# # 执行JS代码
# driver.execute_script("js")
# # JS模拟鼠标滚动
# driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
#
# # 13）窗口截屏
# # 浏览器窗口截屏，参数代表文件保存地址及文件名、格式。只写文件名保存至当前路径，若写路径，则路径必须存在
# driver.get_screenshot_as_file('a.png')
#
# # 14）关闭
# # 关闭网页：关闭当前窗口
# driver.close()
# # 退出整个浏览器会话：关闭所有打开的浏览器窗口，并终止与WebDriver的连接
# driver.quit()

# 15）浏览器自定义配置
# 打开自定义配置
options = webdriver.ChromeOptions()
# 设置浏览器不关闭（解决闪退/自动关闭）
options.add_experimental_option('detach', True)
# 禁用浏览器扩展
options.add_argument('--disable-extensions')
# 禁用浏览器弹窗
options.add_argument('--disable-popup-blocking')
# 设置浏览器UA
options.add_argument('--user-agent=Mozilla/5.0')
# 声明浏览器对象
driver = webdriver.Chrome(options=options)

# 基本示例：

# 打开/关闭浏览器、前进/后退、刷新

# 声明浏览器对象
# driver = webdriver.Chrome()

# 设置浏览器窗口大小
# driver.set_window_size(800, 600)
# 最大化浏览器窗口
driver.maximize_window()

# 浏览器操作
# 等待加载：隐式等待（单位s）
# driver.implicitly_wait(10)

# 访问页面
# 打开百度
driver.get("https://www.hao123.com/")

# 打开CSDN首页
time.sleep(1)          # 暂停1秒
driver.get("https://www.csdn.net/")

time.sleep(1)          # 暂停1秒钟
driver.back()          # 回退：返回上个页面

time.sleep(1)          # 暂停1秒钟
driver.forward()       # 前进：进入下个页面

time.sleep(1)          # 暂停1秒钟
driver.refresh()       # 页面刷新

driver.quit()          # 关闭浏览器

# 3、网页元素定位
# Selenium提供了8种HTML网页元素定位方式，返回单个（element），返回全部（elements）
# 4.x版本及之后请使用find_element(by, value)和find_elements(by, value)方式
'''
id选择器：find_element_by_id()、find_elements_by_id()
class选择器（不支持复合class值）：find_element_by_class_name()、find_elements_by_class_name()
标签名定位：find_element_by_tag_name()、find_elements_by_tag_name()
CSS选择器：find_element_by_css_selector()、find_elements_by_css_selector()
name定位：find_element_by_name()、find_elements_by_name()
XPath定位：find_element_by_xpath()、find_elements_by_xpath()
链接文本定位：find_element_by_link_text()、find_elements_by_link_text()
部分链接文本定位：find_element_by_partial_link_text()、find_elements_by_partial_link_text()
'''
# 1）旧版本（3.x版本及之前）

# # 打开简书
# driver.get("https://www.jianshu.com")
#
# # id选择器
# # 在ID检索搜索框输入文本
# # send_keys()：向目标元素输入数据
# element = driver.find_element_by_id("q").send_keys("xxx")
#
# # class选择器
# # 点击搜索按钮
# # click()：点击操作
# driver.find_element_by_class_name("search-btn").click()
#
# # 清空搜索框关键词
# element.clear()
#
# # 在搜索框输入关键词，并模拟键盘的Enter操作
# element.send_keys("xxx", Keys.ENTER)
#
# # Selenium退出
# # driver.close()：不会清除临时文件夹中的WebDriver临时文件（退出当前标签页）
# # driver.quit()：删除临时文件夹（关闭浏览器）
# driver.close()
#
# # 2）新版本（4.x版本及之后）
#
# from selenium.webdriver.common.by import By
#
# # 打开简书
# driver.get("https://www.jianshu.com")
#
# # id选择器
# # 在ID检索搜索框输入文本
# # send_keys()：向目标元素输入数据
# element = driver.find_element(By.ID, "q")
# time.sleep(1)
# element.send_keys("Python")
#
# # class选择器
# # 点击搜索按钮
# # click()：点击操作
# time.sleep(1)
# driver.find_element(By.CLASS_NAME, "search-btn").click()
#
# # 清空搜索框关键词
# element.clear()
# time.sleep(3)
# # 在搜索框输入关键词，并模拟键盘的Enter操作
# element.send_keys("Java", Keys.ENTER)
#
# # Selenium退出
# # driver.close()：不会清除临时文件夹中的WebDriver临时文件（退出当前标签页）
# # driver.quit()：删除临时文件夹（关闭浏览器）
# driver.close()
#
#
# # 4、操作Cookie
# print(driver.get_cookies())              # 获取Cookie
# driver.add_cookie({'user': 'cookie'})    # 添加Cookie
# print(driver.get_cookie('cookie'))       # 获取设置的Cookie
# driver.delete_cookie('cookie')           # 删除设置的Cookie
# driver.delete_all_cookies()              # 清空所有Cookie
#
# # 5、标签管理
# # 可以在浏览器中切换标签页或增加一个新标签页或删除一个标签页
#
# # 打开百度
# driver.get("https://www.baidu.com")
# time.sleep(1)
#
# # 新增一个标签页
# driver.execute_script('window.open()')
# time.sleep(1)
#
# # 打印标签页
# print(driver.window_handles)
#
# # 切换至标签页1（当前标签页为0）
# driver.switch_to.window(driver.window_handles[1])
# time.sleep(1)
#
# # 在当前标签页访问知乎
# driver.get("https://www.zhihu.com")
# time.sleep(1)
#
# # 退出当前标签页
# driver.close()


# https://zhuanlan.zhihu.com/p/421726412



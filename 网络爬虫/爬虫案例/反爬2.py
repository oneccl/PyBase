
# 滑块验证码

# https://blog.csdn.net/acmman/article/details/133827764
# https://wenku.csdn.net/answer/37mpmi7cxt#:~:text=%23%23%23,%E5%9B%9E%E7%AD%941%EF%BC%9A%20%E4%BD%BF%E7%94%A8selenium%E6%93%8D%E4%BD%9C%E6%BB%91%E5%8A%A8%E9%AA%8C%E8%AF%81%E7%A0%81%E7%9A%84%E6%96%B9%E6%B3%95%E6%98%AF%EF%BC%9A1%E3%80%81%E4%BD%BF%E7%94%A8selenium%E7%9A%84Actions%E7%B1%BB%E7%9A%84drag_and_drop_by_offset%E6%96%B9%E6%B3%95%EF%BC%8C%E6%A8%A1%E6%8B%9F%E7%94%A8%E6%88%B7%E6%8B%96%E5%8A%A8%E6%BB%91%E5%9D%97%EF%BC%9B2%E3%80%81%E4%BD%BF%E7%94%A8selenium%E7%9A%84execute_script%E6%96%B9%E6%B3%95%EF%BC%8C%E6%89%BE%E5%88%B0%E6%BB%91%E5%8A%A8%E6%8C%89%E9%92%AE%EF%BC%8C%E6%A8%A1%E6%8B%9F%E7%94%A8%E6%88%B7%E6%8B%96%E5%8A%A8%E6%BB%91%E5%9D%97%EF%BC%9B3%E3%80%81%E4%BD%BF%E7%94%A8selenium%E7%9A%84execute_script%E6%96%B9%E6%B3%95%EF%BC%8C%E8%AE%A1%E7%AE%97%E5%87%BA%E6%BB%91%E5%8A%A8%E8%B7%9D%E7%A6%BB%EF%BC%8C%E7%84%B6%E5%90%8E%E6%A8%A1%E6%8B%9F%E7%94%A8%E6%88%B7%E6%8B%96%E5%8A%A8%E6%BB%91%E5%9D%97%E3%80%82

# 用例：豆瓣页面登录
# 操作步骤：
"""
1）打开豆瓣登录页面
2）点击页面上的密码登录
3）输入账号密码之后，点击登录豆瓣按钮
4）点击登录后会弹出滑块验证码，拼接验证
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
import time

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

# 最大化浏览器窗口
driver.maximize_window()

# 打开豆瓣登录页
driver.get("https://accounts.douban.com/passport/login")
# 打印页面的标题
print(driver.title)
time.sleep(3)

# 使用浏览器隐式等待3秒
driver.implicitly_wait(3)

# 点击密码登录
driver.find_element(By.XPATH, '//*[@id="account"]/div[2]/div[2]/div/div[1]/ul[1]/li[2]').click()
time.sleep(1)

# 账号密码登录
# 使用浏览器隐式等待3秒
driver.implicitly_wait(3)
# 获取账号密码组件并赋值
user = driver.find_element(By.ID, "username")
user.send_keys("123@qq.com")
time.sleep(1)
pwd = driver.find_element(By.ID, "password")
pwd.send_keys("123456")
time.sleep(1)
# 获取登录按钮并点击登录
login = driver.find_element(By.XPATH, '//*[@id="account"]/div[2]/div[2]/div/div[2]/div[1]/div[4]/a')
login.click()
# 点击登录按钮后，就会出现滑块验证区域，这是一个新增的iframe区域，此时需要将焦点从主页面切换到该iframe区域上，否则将报错：
# selenium.common.exceptions.NoSuchElementException: Message: no such element: Unable to locate element:...


# 滑动验证码

import re
from selenium.webdriver import ActionChains  # 动作类
from selenium.webdriver.support.wait import WebDriverWait  # 等待类
from selenium.webdriver.support import expected_conditions as EC  # 等待条件类
import numpy as np
import urllib.request as req
import cv2

# OpenCV滑块验证码图像缺口识别
from 自动化.图像处理.OpenCV滑块缺口识别 import get_gap_loc


# 切换方法：switch_to.frame()方法，参数为iframe区域唯一属性值（如id值）

# 使用浏览器隐式等待5秒
driver.implicitly_wait(5)
# 切换到弹出的滑块区域（iframe窗口)
driver.switch_to.frame("tcaptcha_iframe_dy")

# 滑块元素
slide = driver.find_element(By.XPATH, '//*[@id="tcOperation"]/div[6]')
# 滑块长度
slide_w = slide.size.get('width')
# 滑块X轴位置
slide_x = slide.location.get('x')

# 轨道元素
track = driver.find_element(By.XPATH, '//*[@id="tcOperation"]/div[7]')
# 轨道长度
track_w = track.size.get('width')

# 最多可滑动距离
# distance = track_w - slide_w - slide_x
# print(f"滑块长度：{slide_w}，轨道长度：{track_w}，滑块位置：{slide_x}，可滑距离：{distance}")
print(f"滑块长度：{slide_w}，轨道长度：{track_w}，滑块位置X：{slide_x}")

# 鼠标事件对象
actionChains = ActionChains(driver)

# 使用浏览器隐式等待5秒
driver.implicitly_wait(5)

# 等待滑块验证图片加载完成
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, 'slideBg')))
# 滑块验证图片
bg_img = driver.find_element(By.XPATH, '//*[@id="slideBg"]')
bg_style = bg_img.get_attribute("style")
# 正则表达式匹配验证图片url，re.S表示点号匹配任意字符，包括换行符
p = r'background-image: url\("(.*?)"\);'
bg_src = re.findall(p, bg_style, re.S)[0]
print(bg_src)

# 计算缺口图像的x轴位置
gap_loc = get_gap_loc(bg_src)
distance = gap_loc[0] - slide_x
print(f"需滑距离：{distance}")

# 按下鼠标左键(按住滑块)
actionChains.click_and_hold(slide).perform()
time.sleep(1)
# 从轨道左滑动到轨道右
# 模拟操作
moved = 0
while moved <= distance / 10:
    moved += 1
    actionChains.move_by_offset(xoffset=moved, yoffset=0).perform()
    time.sleep(0.05)

# 移动完成，松开鼠标
actionChains.release().perform()

# 等待5秒
time.sleep(5)

# 关闭浏览器
driver.quit()



"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/11/5
Time: 14:43
Description:
"""
# 数据来源：国家统计局中国统计年鉴2022年人口数及构成
# 国家统计局：http://www.stats.gov.cn/sj/

# 数据形式：http://www.stats.gov.cn/sj/ndsj/2022/html/C02-01.jpg

# 图像文本识别

import easyocr
import numpy as np
import pandas as pd

# 读取HTTP图像
url = r'http://www.stats.gov.cn/sj/ndsj/2022/html/C02-01.jpg'

# 定义列字段
cols = ['年份', '年末总人口(万)', '男.人口数(万)', '男.比重', '女.人口数(万)', '女.比重', '城镇.人口数(万)', '城镇.比重', '乡村.人口数(万)', '乡村.比重']

# 使用easyocr从图像中提取文本
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
result = reader.readtext(url, detail=0, paragraph=True)

# 选择所需数据范围
# sta_first：起始行第一个数据  end_first：结束行第一个数据
def ocr_data_process(sta_first, end_first):
    # 数据范围
    sta_index = result.index(sta_first)
    end_index = result.index(end_first) + len(cols)
    data_list = result[sta_index: end_index]
    # 处理数据
    data = []
    sta = 0
    end = len(cols)
    while sta <= len(data_list) - 1:
        if end > len(data_list):
            end = len(data_list)
        data.append(data_list[sta: end])
        sta = end
        end += len(cols)
    df = pd.DataFrame(data)
    df.columns = cols
    return df

# 提取2000年到2021年之间的数据
df_res = ocr_data_process('2000', '2021')
print(df_res.to_string())

# 图像灰度化与二值化可以提高识别准确率吗

# OpenCV不支持从HTTP读取图像，使用urllib模块封装如下方法：
import os
import urllib.request as req
import cv2

def cv2_from_url(url):
    # bytearray()：将数据转换为字节数组
    # np.asarray()：复制数据，将结构化数据转换为ndarray
    with req.urlopen(url) as resp:
        # np.uint8：图像压缩
        image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    # cv2.imdecode()：将图像解码为OpenCV图像格式
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

# 图像的灰度化与二值化：
# 需要临时保存的图像文件名及格式
basename = os.path.basename(url)
img = cv2_from_url(url)
# 图像灰度化
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 图像二值化
binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 25)
# 保存图像
cv2.imwrite(basename, binary_img)
# 展示图像
cv2.imshow('binary_img', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 将readtext()方法中的url换成basename，其它保持不动
# 结论：经过灰度化与二值化处理的图像不但没有提高OCR识别的准确率，反而降低了OCR图像识别的准确率
# 对灰度化与二值化后的图像分别做ORC识别，结果同样如此


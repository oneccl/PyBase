

# OpenCV滑动验证码图像缺口位置识别

import cv2
import urllib.request as req
import numpy as np

# 读取图像文件并返回一个image数组表示的图像对象
src1 = r'C:\Users\cc\Desktop\bg.png'
image = cv2.imread(src1)
print("图像大小:", image.shape)

# GaussianBlur()方法用于图像模糊化/降噪操作
# 该方法会基于高斯函数（也称正态分布）创建一个卷积核（也称滤波器），并将该卷积核应用于图像上的每个像素点
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Canny()方法用于图像边缘检测（轮廓）
# image: 输入的单通道灰度图像
# threshold1: 第一个阈值，用于边缘链接。一般设置为较小的值
# threshold2: 第二个阈值，用于边缘链接和强边缘的筛选。一般设置为较大的值
canny = cv2.Canny(blurred, 0, 100)

cv2.imwrite('canny.png', canny)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


# findContours()方法用于检测图像中的轮廓，并返回一个包含所有检测到轮廓的列表
# contours: 可选，输出的轮廓列表。每个轮廓都表示为一个点集
# hierarchy: 可选，输出的轮廓层次结构信息。它描述了轮廓之间的关系，例如父子关系等
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 遍历检测到的所有轮廓的列表
for contour in contours:
    # contourArea()方法用于计算轮廓的面积
    area = cv2.contourArea(contour)
    # arcLength()方法用于计算轮廓的周长或弧长
    length = cv2.arcLength(contour, True)
    # 封闭矩形的面积范围大概是在80*80=6400像素左右，周长是80*4=320像素，给一定的误差范围
    # 如果检测区域面积在5025-7225之间，周长在300-380之间，则是目标区域
    if 5025 < area < 7225 and 300 < length < 380:
        # 计算轮廓的边界矩形，得到坐标和宽高
        # x, y: 边界矩形左上角点的坐标
        # w, h: 边界矩形的宽度和高度
        x, y, w, h = cv2.boundingRect(contour)
        print("缺口位置及大小:", x, y, w, h)
        # 在目标区域上画一个红框检查效果
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite('img.png', img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(x)


# 封装方法

def cv2_from_url(url):
    # bytearray()：将数据转换为字节数组
    # np.asarray()：复制数据，将结构化数据转换为ndarray
    with req.urlopen(url) as resp:
        # np.uint8：图像压缩
        image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    # cv2.imdecode()：将图像解码为OpenCV图像格式
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def get_gap_loc(src: str):
    # 判断是否为URL，获取图像
    image = cv2_from_url(src) if src.startswith('http') else cv2.imread(src)
    # print("图像大小:", image.shape)
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Canny边缘检测
    canny = cv2.Canny(blurred, 0, 100)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gap_list = []
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        # 判断是否为目标区域
        if 5025 < area < 7225 and 300 < length < 380:
            # 获取目标区域的边界框坐标
            x, y, w, h = cv2.boundingRect(contour)
            gap_list.append((x, y, w, h))
    return gap_list[0] if len(set(gap_list)) == 1 else gap_list


src2 = 'https://turing.captcha.qcloud.com/cap_union_new_getcapbysig?img_index=1&image=02790500003332130000000bb5e0481f385d&sess=s0WS8GvssZqdJdsxAWJdnwrfzMiX2nb_lSUPrKho3YmJjHwCWKxs0_EhaT_B-5_h3fijxcI_7zd8qHcOfQDT5ehvlQxFQuTWzUBTbSGNcO-GFP3UgZ0W9Tgu-RtGho1zrwtV_KicSEjbXMC_tnamwH8CKyNRNMLDqgJzIAPgyJVDl39_aaHywQgCnBy0C9FrVDvxBj1PYAo-eFZBNqYnnRk6_Fkf6AD1R7DF34SnuTijXwXJZuXyB7kBYiy0RgGbMQhKDWcjswAxivFtcgUQR45JSjuFb03PzOeEmYx6mROPH8mBzrI3em04iAWYJXjGycrxggZJ-1-FOlVvQTnxurRyoTKUu0EXpn5TMbBttkZeqPa2dSX9PiOA**'

# HTTP图像和保存到本地的图像大小相同，缺口位置识别相同
print(get_gap_loc(src1))
print(get_gap_loc(src2))




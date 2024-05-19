
# OpenCV模块

# OpenCV（Open Source Computer Vision Library）是一个基于BSD许可（开源）发行的跨平台计算机视觉库，主要用于图像和视频处理，可以运行在Linux、Windows、Android和MacOS操作系统上
# OpenCV轻量级且高效：由一系列C函数和少量C++类构成，同时提供了Java、Python、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法
# OpenCV使用C++语言编写，它的主要接口也是C++语言，但是依然保留了大量的C语言接口
# 在计算机视觉项目的开发中，OpenCV作为较大众的开源库，拥有了丰富的常用图像处理函数库，能够快速的实现一些图像处理和识别的任务

# OpenCV官网：https://opencv.org/
# 官方文档参考：https://docs.opencv.org/4.x/

# OpenCV库主要有4个模块：core、imgproc、highgui和videoio
# - core：包含OpenCV库的核心功能，如数据类型、矩阵操作、数组操作、图像处理等
# - imgproc：包含图像处理函数，如阈值处理、滤波、边缘检测、形态学操作、直方图处理等
# - highgui：提供了一些图形界面相关的函数，如图像显示、鼠标和键盘事件处理、视频播放等
# - videoio：提供了一些视频处理相关的函数，如视频的读取和保存、视频的帧率、分辨率等

# OpenCV的应用场景：
# - 物体识别与跟踪：如人脸识别、车牌识别、文本识别、自动驾驶等
# - 图像分割与边缘检测：如医学图像肿瘤分割和边缘检测，以定量诊断和治疗
# - 图像特征提取与描述：如图像拼接和全景重建、深度学习等

# 安装：pip install opencv-python

# 基本使用
import cv2
import numpy as np

# 1）读取图像（不支持HTTP读取）
# cv2.imread(filename, flags)
# - filename：图像文件路径
# - flags：指定图像模式
#   - cv2.IMREAD_COLOR：默认，彩色（忽略alpha通道）模式
#   - cv2.IMREAD_GRAYSCALE：灰度模式
#   - cv2.IMREAD_UNCHANGED：完整图像（包含alpha通道）
img = cv2.imread(r'C:\Users\cc\Desktop\th.jpg')
# 获取图像宽高
width, height, mode = img.shape
print(width, height)

# 2）显示图像
# cv2.imshow(winname, mat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# - winname：图像名称
# - mat：读取的图像对象
# - cv2.waitKey(0)：等待键盘输入（单位：ms），0表示无限等待，没有该操作图像会一闪而逝
# - cv2.destroyAllWindows()：销毁所有窗口
cv2.imshow('th', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3）保存图像
# cv2.imwrite(filename, img, params)
# - filename：保存的图像文件名（带后缀）
# - img：要保存的图像对象
# - params：压缩级别，默认3
cv2.imwrite('save.jpg', img)

# 4）图像缩放
# cv2.resize(src, dsize)
# - src：要缩放的图像
# - dsize：目标大小
resized_img = cv2.resize(img, (200, 200))
cv2.imshow('resized_img', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5）图像裁剪
cropped_img = img[0: 40, 0: 40]
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6）图像旋转
# cv2.getRotationMatrix2D(center, angle, scale)：图像旋转
# center：旋转中心  angle：旋转角度（逆时针） scale：比例
# cv2.warpAffine(src, M, dsize)：图像平移
# src：要旋转的图像  M：矩阵  dsize：旋转后图像大小
M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
rotated_img = cv2.warpAffine(img, M, (width, height))
cv2.imshow('rotated_img', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7）图像颜色空间转换
# cv2.COLOR_BGR2GRAY：图像灰度化
# cv2.COLOR_BGR2HSV：RGB转HSV
# cv2.cvtColor(src, code)
# src：源图像  code：转换码
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_img', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8）图像平滑、锐化处理（降噪）
# cv2.blur()：均值滤波
# cv2.GaussianBlur()：高斯滤波
# cv2.medianBlur()：中值滤波
# cv2.bilateralFilter()：双边滤波
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('blurred_img', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 9）图像添加文字
# cv2.putText(img,text,org,fontFace,fontScale,color,thickness)
# 参数依次：图像，文本，左上角坐标，字体类型，字体大小，字体颜色，字体粗细
# cv2不支持中文文字，需要使用Pillow库
text_img = cv2.putText(img, '文本', (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
cv2.imshow('text_img', text_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 10）图像融合（几何运算）
# cv2.addWeight(src1,alpha,src2,beta,gamma)
# src1：图1  alpha：图1权重  src2：图2  beta：图2权重，beta=1-alpha  gamma：重叠偏置，一般为0
# 函数原型：out = src1*alpha+src2*(1-alpha)+gamma
# 当alpha和beta都为1时，相当于cv2.add()
img1 = resized_img[0: 100, 0: 100]
img2 = resized_img[100: 200, 100: 200]
img3 = cv2.add(img1, img2)
img4 = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 11）边缘检测
# cv2.Canny(img, threshold1, threshold2)
# img：图像  threshold1：最低阈值，确定潜在边缘  threshold2：最高阈值，确定真正边缘
edges_img = cv2.Canny(img, 100, 200)
cv2.imshow('edges_img', edges_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 12）阈值分割
# 图像二值化
# 1）全局阈值二值化
# cv2.threshold(src, thresh, maxval, type)
# src：图像  thresh：最低阈值  maxval：最高阈值  type：二值化类型
# 以灰度模式读取图像
img_g = cv2.imread(r'C:\Users\cc\Desktop\th.jpg', 0)
retval, binary = cv2.threshold(img_g, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 2）局部（自适应）阈值二值化
# cv2.adaptiveThreshold(src,maxValue,adaptiveMethod,thresholdType,blockSize,C)
# src：图像 maxValue：灰度值 adaptiveMethod：自适应阈值算法 thresholdType：二值化类型 blockSize：邻域大小，分割区域大小，取奇数
# C：阈值调整常数，每个区域的阈值的减去该常数作为该区域的最终阈值，可以为负数
binary_img = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 25)
cv2.imshow('binary_img', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 13）图像数学形态学
# 图像腐蚀与膨胀
# 腐蚀和膨胀是针对白色部分（高亮部分）而言的。膨胀就是对图像高亮部分进行“领域扩张”，效果图拥有比原图更大的高亮区域。腐蚀是原图中的高亮区域被蚕食，效果图拥有比原图更小的高亮区域
# 膨胀用来处理缺陷问题；腐蚀用来处理毛刺问题
# 1）膨胀
# cv2.dilate(src,kernel,iterations)
# src：图像 kernel：结构元及卷积核数 iterations：迭代次数越多，膨胀效果越强
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
cv2.imshow('dilated_img', dilated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 2）腐蚀
# cv2.erode(src,kernel,iterations)
# src：图像 kernel：结构元(形状)及卷积核数 iterations：迭代次数越多，腐蚀效果越强
eroded_img = cv2.erode(dilated_img, kernel, iterations=3)
cv2.imshow('eroded_img', eroded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# PIL.Image与OpenCV格式转换
from PIL import Image

# 1）Image.open()转cv2.imread()
pil_img = Image.open(r'C:\Users\cc\Desktop\th.jpg')
cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
cv2.imshow('cv2_img', cv2_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 2）cv2.imread()转Image.open()
img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
img_pil.show()

# 图像识别应用案例
# 使用OpenCV的文字识别算法，可以对图像中的人脸、物体、文字等进行识别，例如识别身份证号码、验证码等
# 1）人脸识别
import cv2

# 加载预训练的人脸级联分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 读取图像
img = cv2.imread('face.jpg')
# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用级联分类器检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# 为每个检测到的人脸绘制一个矩形（标记人脸）
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# 显示结果
cv2.imshow('Faces found', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 实时人脸识别
import cv2

# 加载预训练的人脸级联分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 打开摄像头
cap = cv2.VideoCapture(0)
while True:
    # 读取一帧
    ret, frame = cap.read()
    # 将帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用级联分类器检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 为每个检测到的人脸绘制一个矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 显示结果
    cv2.imshow('Faces found', frame)
    # 按q退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()

# 2）车牌识别
import cv2
import pytesseract

# 读取图像
img = cv2.imread('car_plate.jpg')
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 进行高斯滤波
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 进行边缘检测
edges = cv2.Canny(blur, 100, 200)
# 查找轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 对轮廓进行筛选
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000 and area < 50000:
        x, y, w, h = cv2.boundingRect(contour)
        if w / h > 2 and w / h < 6:
            # 截取车牌区域
            plate = gray[y:y+h, x:x+w]
            # 进行二值化处理
            ret, thresh = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # 识别车牌号码
            plate_number = pytesseract.image_to_string(thresh, lang='chi_sim')
            print('车牌号码：', plate_number)
 # 展示结果
cv2.imshow('Original Image', img)
cv2.imshow('Gray Image', gray)
cv2.imshow('Edges Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3）文本识别
import cv2
import pytesseract

# 读取身份证图片
img = cv2.imread('id_card.jpg')
# 将图片转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对灰度图像进行二值化处理
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 对二值化图像进行膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilation = cv2.dilate(thresh, kernel, iterations=1)
# 在膨胀后的图像中查找轮廓
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 遍历轮廓，找到身份证号码所在的矩形区域
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 100 and h > 20 and w < 200 and h < 50:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, lang='chi_sim')
        print('身份证号码: ', text)
# 显示识别结果
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 通用图像文本识别：灰度化->二值化->降噪

def get_img_text(src: str):
    img = cv2_from_url(src) if src.startswith('http') else cv2.imread(src)
    # 图像灰度化
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 图像二值化
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 25)
    # 降噪
    blurred_img = cv2.GaussianBlur(binary_img, (5, 5), 0)
    # 使用PyTesseract库识别图像文本
    text = pytesseract.image_to_string(blurred_img, lang='chi_sim')
    return text





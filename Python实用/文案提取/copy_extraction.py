"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2024/7/7
Time: 22:24
Description:
"""

# 视频文案提取
# https://www.jb51.net/python/315928ury.htm

# 在多媒体应用中，视频是一个信息量巨大的载体。然而，有时我们需要从视频中提取语音并转换为文本，以用于文本分析和机器学习训练
# Python实现视频转音频和音频转文字的功能主要有两个库：
# moviepy：用于将视频转为音频
# SpeechRecognition：用于将音频转换为文字
# 安装：
# pip install moviepy
# pip install SpeechRecognition

# 文案提取分为两步：视频转音频、音频转文字

# 1、视频转音频
from moviepy.editor import VideoFileClip

# 选择视频文件
# 视频文件路径或文件名
video_path = r"C:\Users\cc\Desktop\02dc5796f5fc501549774efa2520831f.mp4"

# 使用VideoFileClip函数创建一个VideoFileClip对象，用于处理视频文件
video = VideoFileClip(video_path)

# 使用audio方法从VideoFileClip对象中提取音频
audio = video.audio

# 使用write_audiofile方法将提取的音频保存到文件中
# 音频文件输出路径或文件名
audio_output_path = "audio.wav"
audio.write_audiofile(audio_output_path)

# 2、音频转文字
import speech_recognition as sr
import os

# 选择音频文件
# 音频文件路径或文件名
audio_path = "audio.wav"

# 创建Recognizer对象，用于处理音频文件
recognizer = sr.Recognizer()

# 使用Recognizer对象的record方法读取音频文件
with sr.AudioFile(audio_path) as source:
    audio = recognizer.record(source)

# 语音识别
# recognize_google函数可能不会在所有音频文件上工作，因为它依赖于云服务或本地语音识别引擎的准确性和性能。对于特定的应用，可能需要对音频进行预处理，例如降噪或调整录音条件以提高识别准确率
text = recognizer.recognize_google(audio, language='zh-CN')
print(text)

# 优化
# try:
#     # 使用Recognizer对象的recognize_google方法将音频转换为文字
#     # 使用Google Cloud Speech-to-Text API：将音频内容发送到Google的免费语音识别服务，并尝试将其转换为文本
#     text = recognizer.recognize_google(audio, language='zh-CN')
#     # 转换结果
#     print(text)
# except sr.RequestError:
#     print("API请求失败!")
# except sr.UnknownValueError:
#     print("语音无法识别!")
# finally:
#     ...
    # 清理临时文件
    # os.remove(audio_path)



# 封装
# 视频转音频
def video_to_audio(video_path, audio_output_path):
    # 创建VideoFileClip对象
    video = VideoFileClip(video_path)

    # 提取音频
    audio = video.audio

    # 保存音频文件
    audio.write_audiofile(audio_output_path)


# 音频转文字
def audio_to_text(audio_path):
    # 创建Recognizer对象
    recognizer = sr.Recognizer()

    # 读取音频文件
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    # 将音频转换为文字
    text = recognizer.recognize_google(audio)

    # 打印转换结果
    print(text)


# # 视频文件路径或文件名
# video_path = "video.mp4"
# # 音频文件输出路径或文件名
# audio_output_path = "audio.wav"
#
# # 视频转音频
# video_to_audio(video_path, audio_output_path)
# # 音频转文字
# audio_to_text(audio_output_path)




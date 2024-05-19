
# re模块补充：

import re

# 判断字符串是否以数字或指定正则开头

# s1 = "10.-Not All"
# s2 = "-1.-Not All"

# print(s1[0].isdigit())       # True
# print(s1[0:2].isdigit())     # True
#
# print(re.match('\\d', s1).group(0))               # 1
# print(True if re.match('\\d', s1) else False)     # True
# print(True if re.match('-\\d', s2) else False)    # True


# 使用正则匹配整个字符串

# s1 = 'MENA_10'
# s2 = 'MENA_10a'
# s3 = '(G311; G310)'
# s4 = '(G311); (G310)'
# # 匹配整个字符串以大写字母数字下划线组成（开头和结尾）
# print(True if re.fullmatch('^[A-Z0-9_]+$', s1) else False)  # True
# print(re.fullmatch('^[A-Z0-9_]+$', s1).group(0))            # MENA_10
# # 使用flags=re.IGNORECASE忽略字符串中的大小写进行匹配
# print(True if re.fullmatch('^[A-Z0-9_]+$', s2, flags=re.IGNORECASE) else False)  # True
# print(re.fullmatch('^[A-Z0-9_]+$', s2, flags=re.IGNORECASE).group(0))            # MENA_10a
# # 匹配整个字符串以大写字母开头，以小写字母或数字结尾，中间任意
# print(True if re.fullmatch('^[A-Z].*[a-z0-9]$', s2) else False)  # True
# print(re.fullmatch('^[A-Z].*[a-z0-9]$', s2).group(0))            # MENA_10a
# # 匹配整个字符串以()结尾，()中固定元字符为大写字母、任意0或多个字符、数字
# print(True if re.fullmatch('\\([A-Z]+.*[0-9]+\\)', s3) else False)  # True
# print(re.fullmatch('\\([A-Z]+.*[0-9]+\\)', s3).group(0))            # (G311; G310)
# print(True if re.fullmatch('\\([A-Z]+.*[0-9]+\\)', s4) else False)  # True
# print(re.fullmatch('\\([A-Z]+.*[0-9]+\\)', s4).group(0))            # (G311); (G310)
# # 匹配字符串开头以()结尾，()中固定元字符为大写字母、任意一个字符、数字
# print(True if re.match('\\([A-Z]+.[0-9]+\\)', s4) else False)  # True
# print(re.match('\\([A-Z]+.[0-9]+\\)', s4).group(0))            # (G311)



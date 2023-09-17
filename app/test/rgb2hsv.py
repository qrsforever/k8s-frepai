import cv2
import numpy as np

# http://www.jiniannet.com/Page/allcolor
rgb = '#9395A1,#7A8082,#86949E,#698290,#7D97A4,#716D72,#525D71'

rgb = rgb.split(',')

# 转换为BGR格式，并将16进制转换为10进制
bgr = [[int(r[5:7], 16), int(r[3:5], 16), int(r[1:3], 16)] for r in rgb]

# 转换为HSV格式
for b in bgr:
    print(b)
# hsv = [list(cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2HSV)[0][0]) for b in bgr]
# 
# hsv = np.array(hsv)
# print('H:', min(hsv[:, 0]), max(hsv[:, 0]))
# print('S:', min(hsv[:, 1]), max(hsv[:, 1]))
# print('V:', min(hsv[:, 2]), max(hsv[:, 2]))

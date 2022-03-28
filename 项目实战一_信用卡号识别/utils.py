import os
import numpy as np
from imutils import contours
import cv2
import pickle


def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 下面将轮廓进行排序，这是因为必须保证轮廓的顺序是0-9的顺序排列着
def sort_contours(cnts, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小矩形，把找到的形状包起来x,y,h,w
    
    # 根据每个轮廓左上角的点进行排序， 这样能保证轮廓的顺序就是0-9的数字排列顺序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda x:x[1][i], reverse=reverse))
    
    return cnts, boundingBoxes 


def credit_process(credit_gray):
    
    # 顶帽操作，突出更明亮的区域
    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 自定义卷积核的大小了
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(credit_gray, cv2.MORPH_TOPHAT, rectKernel)
    
    # 水平边缘检测  
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 水平边缘检测
    # gradX = cv2.convertScaleAbs(gradX)    这个操作会把一些背景边缘也给检测出来，加了一些噪声

    # 所以下面手动归一化操作
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX-minVal) / (maxVal-minVal)))
    gradX = gradX.astype('uint8')
    
    # 闭操作: 先膨胀， 后腐蚀  膨胀就能连成一块了
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    #THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0  让opencv自动的去做判断，找合适的阈值，这样就能自动找出哪些有用，哪些没用
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
    
    #再来一个闭操作
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
    
    return thresh

def comput_contours(thresh):
    
    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    
    # 找到包围数字的那四个大轮廓
    locs = []
    # 遍历轮廓
    for i, c in enumerate(cnts):
        # 计算外接矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 选择合适的区域， 这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4.0:
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # 符合
                locs.append((x, y, w, h))

    # 轮廓从左到右排序
    locs = sorted(locs, key=lambda x: x[0])
    return locs

def getOutput(locs, digits2Cnt, credit_card, credit_gray):
    outputs = []

    # 遍历每一个轮廓中的的数字
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # 初始化组
        groupOutput = []

        # 根据坐标提取每一组
        group = credit_gray[gY-5:gY+gH+5, gX-5:gX+gW+5]  # 有5的一个容错长度

        # 对于这每一组，先预处理  
        # 二值化，自动寻找合适阈值，增强对比，更突出有用的部分，即数字
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 计算每一组的轮廓
        digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = sort_contours(digitCnts, method='left-to-right')[0]

        # 拿到每一组的每一个数字，然后进行模板匹配
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y+h, x:x+w]
            roi = cv2.resize(roi, (57, 88))

            # 模板匹配
            scores = []
            for (digit, digitROI) in digits2Cnt.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            # 得到合适的数字
            # 这是个列表，存储的每个小组里面的数字识别结果
            groupOutput.append(str(np.argmax(scores)))

        # 画出来
        cv2.rectangle(credit_card, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(credit_card, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 合并到最后的结果里面
        outputs.extend(groupOutput)
    return outputs
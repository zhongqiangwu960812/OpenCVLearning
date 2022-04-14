import numpy as np
import cv2

def cv_imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(pts):
    # 一共四个点
    rect = np.zeros((4, 2), dtype='float32')
    
    # 按照顺序对应坐标0123分别是左上， 右上， 右下， 左下
    s = pts.sum(axis=1)  # 横纵坐标相加， 左上的之和最小，右下的之和最大
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)   # 纵坐标-横坐标， 右上的之差最小， 左下的之差最大
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换函数
def four_point_transform(img, pts):
    # 获取坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算输入的w和h的值，方便定位新坐标
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 变换后对应坐标位置
    dst = np.array([
        [0, 0], 
        [maxWidth-1, 0], 
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype='float32')
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    return warped

def sort_contours(cnts, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def get_dotCnts(cnts):
    # 这里对上面的轮廓进行一次筛选， 确保是答题卡的轮廓，因为有可能检测到里面的原形轮廓，另外一个就是最外面的答题卡轮廓目前并不是
    # 很标准， 边缘是锯齿形的，所以要在看是否是最外面的轮廓之前，进行轮廓近似操作
    if len(cnts) > 0:
        # 根据轮廓大小进行排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for c in cnts:
            # 轮廓近似
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            
            # 保存四个顶点，为透视变换做准备
            if len(approx) == 4:
                dotCnt = approx
                break
        
        return dotCnt

def filter_cnts(cnts):
    # 此时检测到的轮廓很多，我们下面需要过滤， 选择出答案的那些圆形轮廓
    questionCnts = []
    for c in cnts:
        # 计算比例和大小
        (x, y, w, h) = cv2.boundingRect(c)   # 外接矩形， 原型的外接矩形的长宽比例接近1
        ar = w / float(h)
        
        # 根据实际情况指定标准
        if w >= 20 and h >= 20 and ar >= 0.0 and ar <= 1.1:
            questionCnts.append(c)

    return questionCnts


def get_dect_res(questionCnts, ANSWER_KEY, warped, thresh):

    correct = 0

    # 遍历每个题目
    for (q_idx, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # 拿到当前题目的5个选项，并且从左到右排序
        cnts = sort_contours(questionCnts[i:i+5])[0]
        selected = None
        
        # 遍历每个结果
        for (j, c) in enumerate(cnts):
            # 使用mask来判断选择的是哪个答案
            mask = np.zeros(thresh.shape, dtype='uint8')
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # 通过计算非零点数量来算是否选择了当前答案
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total_non_zeros = cv2.countNonZero(mask)
            
            # 通过阈值判断  选出非零点数量最多的那个来
            if selected is None or total_non_zeros > selected[0]:
                selected = (total_non_zeros, j)
        
        # 对比正确答案
        color = (255, 0, 0)
        true_ans = ANSWER_KEY[q_idx]
        
        # 选择正确
        if true_ans == selected[1]:
            correct += 1
            color = (0, 255, 0)
        
        # 绘图
        cv2.drawContours(warped, [cnts[true_ans]], -1, color, 3)

    return correct, warped
import numpy as np
import cv2
from utils import four_point_transform, sort_contours, get_dotCnts, filter_cnts, get_dect_res


def answer_dect(img, ANSWER_KEY):

    # 转成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    blured = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edged = cv2.Canny(blured, 75, 200)

    # 轮廓检测
    # 轮廓检测这里应该在边缘检测的结果上进行，才能锁定答题区域， 如果换成灰度图，这里检测不到答题卡的轮廓
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取答题卡边框的四个顶点坐标
    dotCnts = get_dotCnts(cnts)

    # 透视变换
    warped = four_point_transform(gray, dotCnts.reshape(4, 2))

    # 圆圈的轮廓检测
    # 在轮廓检测之前，先通过阈值把图像处理成黑白图像，这样后面找圆圈的轮廓才能更加清晰
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 检测每个圆圈轮廓
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓
    questionCnts = filter_cnts(cnts)

    # 接下来， 就是把这些圆圈排序，首先需要先按照每个题排列好，不同题的x坐标一致， y坐标是从小到大
    questionCnts = sort_contours(questionCnts, method='top-to-bottom')[0]

    # 遍历每一题的每个圆圈，获取最终的结果及图像
    correct, warped = get_dect_res(questionCnts, ANSWER_KEY, warped, thresh)


    # 结果可视化
    exam_img = warped.copy()
    score = (correct / 5) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(exam_img, "{:.2f}%".format(score), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return exam_img



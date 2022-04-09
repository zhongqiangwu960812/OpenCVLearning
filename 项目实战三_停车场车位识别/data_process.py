import cv2
import pickle

from utils.data_utils import select_rgb_white_yellow, select_region, hough_lines, identity_blocks, cv_imshow, rect_finetune, draw_parking, save_images_for_cnn

# 给定一帧图片，经过数据预处理操作，生成CNN的预测测试集
def data_process(test_image, low_threshold=50, high_threshold=200, save_cnn_data=False):

    # 去掉背景图片
    masked_img = select_rgb_white_yellow(test_image)

    # 转成灰度图
    gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Canny 边缘检测
    edges_img = cv2.Canny(gray_img, low_threshold, high_threshold)

    # 停车场区域标定
    roi_image = select_region(edges_img)

    # 霍夫变换找到直线
    list_of_lines = hough_lines(roi_image)

    # 检测每列停车位，画矩形框
    new_image, rects = identity_blocks(test_image, list_of_lines)
    # cv_imshow('new_image', new_image)

    # 矩形框微调
    new_image, rects = rect_finetune(test_image, rects)

    # 框定停车位
    new_image, spot_dict = draw_parking(test_image, rects)
    # cv_imshow('new_image', new_image)

    # 去掉多余的停车位
    invalid_spots = [10, 11, 33, 34, 37, 38, 61, 62, 93, 94, 95, 97, 98, 135, 137, 138, 187, 249,
                     250, 253, 254, 323, 324, 327, 328, 467, 468, 531, 532]
    valid_spots_dict = {}
    cur_idx = 1
    for k, v in spot_dict.items():
        if v in invalid_spots:
            continue
        valid_spots_dict[k] = cur_idx
        cur_idx += 1

    # 保存这个停车位字典
    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(valid_spots_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 这句话其实没有用，只要有这个有效的停车点字典就可以给定图片现场切分停车位，然后模型预测，不用先保存起来，然后读取，这样反而效率很低
    if save_cnn_data:
        save_images_for_cnn(test_image, valid_spots_dict)

    return valid_spots_dict













import cv2
import numpy as np
from predict import model_infer
from PIL import Image
from tqdm import tqdm

def predict_on_img(img, spot_dict, model, class_indict, make_copy=True, color=[0, 255, 0], alpha=0.5, save=True):
    # 这个是停车场的全景图像
    if make_copy:
        new_image = np.copy(img)
        overlay = np.copy(img)

    cnt_empty, all_spots = 0, 0
    for spot in tqdm(spot_dict.keys()):
        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        spot_img = img[y1:y2, x1:x2]
        spot_img_pil = Image.fromarray(spot_img)

        label = model_infer(spot_img_pil, model, class_indict)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cnt_empty += 1

    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

    # 显示结果的
    cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    if save:
        filename = 'with_marking_predict.jpg'
        cv2.imwrite(filename, new_image)
    # cv_imshow('new_image', new_image)
    return new_image



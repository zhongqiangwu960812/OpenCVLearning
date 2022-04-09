import cv2
from predict_on_spot_img import predict_on_img

def predict_on_video(video_path, spot_dict, model, class_indict, ret=True):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while ret:
        ret, image = cap.read()
        count += 1

        if count == 5:
            count = 0
            new_image = predict_on_img(image, spot_dict, model, class_indict, save=False)

            cv2.imshow('frame', new_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()

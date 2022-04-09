import torch
import os
# golb模块是用来查找符合特定规则命名的文件名的“路径+文件名”，其功能就是检索路径
import glob
import cv2
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt

from data_process import data_process
from train import train_model
from torchvision.models import resnet34
from predict_on_spot_img import predict_on_img
from predict_on_spot_video import predict_on_video

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # 如果没有停车位字典，先生成字典
    if not os.path.exists('spot_dict.pickle'):
        # 读入数据
        test_images = [cv2.imread(path) for path in glob.glob('test_images/*.jpg')]  # BGR
        spot_dict = data_process(test_images[1])
    else:
        spot_dict = pickle.load(open('spot_dict.pickle', 'rb'))

    # 读取训练好的ResNet模型
    if not os.path.exists('saved_model_weight/resnet34_pretrain.pth'):
        # 调用train函数训练模型
        train_model()
    else:
        # read class_indict
        json_path = './idx2class.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # 导入与训练好的模型
        model = resnet34(num_classes=2).to(device)
        weights_path = "saved_model_weight/resnet34_pretrain.pth"
        model.load_state_dict(torch.load(weights_path))

        # 预测一张停车场的图片
        #img = plt.imread('test_images/scene1380.jpg')
        #predict_on_img(img, spot_dict, model, class_indict)

        # 预测parking_video.mp4
        video_path = 'video/parking_video.mp4'
        predict_on_video(video_path, spot_dict, model, class_indict)



"""
这个文件扩展数据集用， 目前train_data数据量还是太少，所以就打算把cnn_pred_data下面的数据，用模型预测，然后剪切到模型训练的相应目录下进行扩增

这个函数，以后就接受一张停车场的图片， 视频的一帧， 然后保存到test_images目录下面

然后从这个里面读取图片，然后通过数据预处理操作，把停车位划分开， 然后保存到cnn_pred_data里面，然后再从这里面读数据，预测， 剪切到train_data/train中

这样，数据量大了，能提高模型的预测准确度， 因为发现目前模型预测的并不是很准
"""
import os
import json

import cv2
import torch
import glob
import matplotlib.pyplot as plt
from shutil import copy, move
from PIL import Image
from torchvision.models import resnet34
from predict import model_infer
from data_process import data_process
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    #test_image = cv2.imread('test_images/scene1410.jpg')
    #_ = data_process(test_image, save_cnn_data=True)

    # read class_indict
    json_path = './idx2class.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 导入与训练好的模型
    model = resnet34(num_classes=2).to(device)
    weights_path = "saved_model_weight/resnet34_pretrain.pth"
    model.load_state_dict(torch.load(weights_path))

    test_images_dict = {path: Image.open(path) for path in glob.glob('cnn_pred_data/*.jpg')}  # BGR

    for img_path, img in tqdm(test_images_dict.items()):
        label = model_infer(img, model, class_indict)

        if label == 'empty':
            # 把这张img复制到train_data/train/empty下
            new_path = os.path.join('train_data', 'train', 'empty')
        else:
            # 把这张img复制到train_data/train/occupied下
            new_path = os.path.join('train_data', 'train', 'occupied')

        move(img_path, new_path)









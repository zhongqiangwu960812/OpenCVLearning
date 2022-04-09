import os
import json
import torch
import glob
import matplotlib.pyplot as plt
from shutil import copy, rmtree
from PIL import Image
from torchvision.models import resnet34
from utils.model_utils import data_transform_pretrain, model_predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_infer(img, model, class_indict):

    # [N, C, H, W]
    img = data_transform_pretrain["test"](img)
    img = torch.unsqueeze(img, dim=0)

    # 模型预测
    predict, predict_cla = model_predict(model, img, device)

    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))

    # print("\npredict_res: {}".format(class_indict[str(predict_cla)]))

    predict_res = class_indict[str(predict_cla)]
    return predict_res



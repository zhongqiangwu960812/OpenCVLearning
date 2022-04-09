import os
import torch
import torch.nn as nn
from torchvision.models import resnet34
import torch.optim as optim

from utils.model_utils import get_dataloader, data_transform_pretrain, model_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model():
    # 获取dataloader
    data_root = os.getcwd()
    image_path = os.path.join(data_root, "train_data")
    train_data_path = os.path.join(image_path, "train")
    val_data_path = os.path.join(image_path, "test")
    train_loader, validat_loader, train_num, val_num = get_dataloader(train_data_path, val_data_path,
                                                                      data_transform_pretrain, batch_size=8)

    # 创建模型 注意这里没指定类的个数，默认是1000类
    net = resnet34()
    model_weight_path = 'saved_model_weight/resnet34_pretrain_ori_low_torch_version.pth'

    # 使用预训练的参数，然后进行finetune
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # 改变fc layer structure  把fc的输出维度改为2
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)

    # 模型训练配置
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    save_path = "saved_model_weight/resnet34_pretrain.pth"
    best_acc = 0.
    train_steps = len(train_loader)

    model_train(net, train_loader, validat_loader, epochs, device, optimizer, loss_function, train_steps, val_num,
                save_path, best_acc)


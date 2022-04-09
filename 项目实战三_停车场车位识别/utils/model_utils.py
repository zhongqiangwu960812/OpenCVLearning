import os
import sys
import json
import torch
from torchvision import transforms, datasets, utils
from tqdm import tqdm


data_transform_pretrain = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 对图像随机裁剪， 训练集用，验证集不用
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 这里的中心化处理参数需要官方给定的参数，这里是ImageNet图片的各个通道的均值和方差，不能随意指定了
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "val": transforms.Compose([
            # 验证过程中，这里也进行了一点点改动
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

def get_dataloader(train_root, val_root, transform, batch_size, shuffle=True):
    """
    :param root: file_path
    :param transform: image_process transform
    :param batch_size: batch_size
    :param shuffle: bool, shuffle
    :return: train_loader, validate_loader
    """
    # 先构建dataset
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=val_root, transform=transform['val'])
    train_num, val_num = len(train_dataset), len(val_dataset)

    # 这里要保存一份idx到具体类别的映射，预测的时候会用到
    if not os.path.exists('idx2class.json'):
        # {'empty':0, 'occupied':1}  标签-索引，
        class2idx = train_dataset.class_to_idx
        idx2class = {val: idx for idx, val in class2idx.items()}
        # 保存成json， 预测的时候用
        json_str = json.dumps(idx2class, indent=4)
        with open('idx2class.json', 'w') as json_file:
            json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers  Windows系统下必须是0 否则报BrokenPipeError: [Errno 32] Broken pipe
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    return train_loader, validate_loader, train_num, val_num


def model_train(model, train_loader, validate_loader, epochs, device, optimizer, loss_function,
          train_steps, val_num, save_path, best_acc=0., model_name='common'):

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            if model_name == 'common':
                outputs = model(images)
                loss = loss_function(outputs, labels)
            elif model_name == 'googlenet':
                logits, aux_logits2, aux_logits1 = model(images)
                loss0 = loss_function(logits, labels)
                loss1 = loss_function(aux_logits2, labels)
                loss2 = loss_function(aux_logits1, labels)
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3

            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        model.eval()   # dorpout不会被使用, aux_loss不会使用
        acc = 0.
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                if torch.cuda.is_available():
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                outputs = model(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 保存最优参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

def model_predict(model, img, device):
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    return predict, predict_cla
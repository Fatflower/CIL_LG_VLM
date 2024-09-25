from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np

import json
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]



class CIFAR100_I2T(iData):
    '''
    Dataset Name:   CIFAR-100 dataset (Canadian Institute for Advanced Research, 100 classes)
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.RandomCrop(224, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 10)),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=63/255)
        ]
        self.test_trsf = []
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size), interpolation=BICUBIC),
            transforms.CenterCrop((self.img_size, self.img_size)), 
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        self.class_order = np.arange(100).tolist()

    def getdata(self, file_dir):
        class_names = os.listdir(file_dir)
        class_names.sort()
        data = []
        targets = []
        # id_des = []
        # ood_des = []
        # if ID_des_dir:
        #     with open('ID_des_dir') as f:
        #         ID_text = json.load(f)
        # if OOD_des_dir:
        #     with open('dataset/I2T_CIFAR100.json') as f:
        #         OOD_text = json.load(f)    

        for class_name in class_names:
            class_path = os.path.join(file_dir, class_name)
            imgs_list = os.listdir(class_path)
            imgs_list.sort()
            for img in imgs_list:
                img_path = os.path.join(class_path, img)
                data.append(img_path)
                targets.append(class_names.index(class_name))
                
        return np.array(data), np.array(targets)

    def download_data(self):
        data_path = "data"
        # data_path = "/home/2021/wentao/Storage/tmp/CLIP_OOD_detection_CIL/data"
        base_dir = os.path.join(data_path,"cifar100_images")
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "val")
        # train_ID_des_dir = os.path.join(data_path, "I2T_CIFAR100_chatgpt.json")
        # train_OOD_des_dir = os.path.join(data_path, "I2T_CIFAR100_near_OOD_chatgpt.json")

        
        self.train_data, self.train_targets = self.getdata(train_dir)
        self.test_data, self.test_targets = self.getdata(test_dir)
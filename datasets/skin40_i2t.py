from torchvision import transforms
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

class Skin40_I2T(iData):
    '''
    Dataset Name:   Skin8 (ISIC_2019_Classification)
    Task:           Skin disease classification
    Data Format:    600x450 color images.
    Data Amount:    3555 for training, 705 for validationg/testing
    Class Num:      8
    Notes:          balanced each sample num of each class

    Reference:      
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.Resize((self.img_size, self.img_size), interpolation=BICUBIC),
            transforms.RandomCrop(224, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 10)),
            # transforms.RandomResizedCrop(224, (0.8, 1)),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size), interpolation=BICUBIC),
            transforms.CenterCrop((self.img_size, self.img_size)), 
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        self.class_order = np.arange(40).tolist()
    
    def getdata(self, fn, img_dir):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(img_dir, temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        # data_path = "/home/zhangwentao/Dataset2"
        train_dir = os.path.join(os.environ["DATA"], "SD-198/main_classes_split/train_1.txt")
        test_dir = os.path.join(os.environ["DATA"], "SD-198/main_classes_split/val_1.txt")
        img_dir = os.path.join(os.environ["DATA"], 'SD-198/images')

        self.train_data, self.train_targets = self.getdata(train_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, img_dir)
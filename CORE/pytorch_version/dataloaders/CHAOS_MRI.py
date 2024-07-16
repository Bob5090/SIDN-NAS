import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import PIL.Image as Image
import os
import torch


class MRIDataset(Dataset):
    def __init__(self, root, data_name, batch_size = 1, transform=None):
        assert data_name in ["Left_Kidney", "Right_Kidney", "Liver", "Spleen"]
        self.transform = transform
        self.x, self.y = self.load_data(root, data_name)
        # print('self.transform', self.transform)
        self.CLASS_WEIGHTS = torch.tensor([1., 1.])  # use cpu

    def __getitem__(self, index):
        # if self.transform is not None:
        #     x, y = self.transform(images=self.x[None, index], segmentation_maps=self.y[None, index])
        # else:
        #     x, y = self.x[None, index], self.y[None, index]

        x, y = self.transform(images=self.x[None, index], segmentation_maps=self.y[None, index])
        x, y = x.astype('float32')[0] / 255., y.astype('float32')[0] / 255.
        x, y = ToTensor()(x), ToTensor()(y)

        ret = {}
        ret['x'] = x.float()
        ret['y'] = y.float()
        return ret

    def __len__(self):
        return self.x.shape[0]

    def load_data(self, path, data_name):
        print('loading ' + data_name + ' data...')
        print('data path: ', path)
        imgs_list = []
        masks_list = []

        ori_path = os.path.join(path, "x")
        ground_path = os.path.join(path, "y_" + data_name)
        names = os.listdir(ori_path)
        n = len(names)
        for i in range(n):
            img = os.path.join(ori_path, names[i])
            mask = os.path.join(ground_path, names[i])
            img = Image.open(img).convert('L')
            mask = Image.open(mask).convert('L')
            imgs_list.append(np.array(img))
            masks_list.append(np.array(mask))

        # print(len(imgs_list))
        # print(len(masks_list))
        # for mask in masks_list:
        #     print(mask.shape)

        imgs_np = np.asarray(imgs_list)
        masks_np = np.asarray(masks_list)
        x = np.asarray(imgs_np, dtype=np.uint8)
        y = np.asarray(masks_np, dtype=np.uint8)
        if len(y.shape) == 3:
            y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        print("Successfully loaded data from " + path)
        print("data shape:", x.shape, y.shape)
        return x, y
"""
#!/usr/bin/python3
-*- coding: utf-8 -*-
@Time    : 2021/9/2 下午4:46
@Author  : xsbank
@Email   : yangsai1991@163.com

"""
import json
import os
import random
import cv2
import torch
import numpy as np
from skimage.util import random_noise
from torch.utils.data import Dataset


class ImageDatasetImg(Dataset):
    def __init__(self, data_root, dataset, uid, express='Micro', augment=False,  mode='train', K=6):
        self.data_root = data_root
        self.length = K
        self.augment = augment
        self.img_list = []
        self.label_list = []

        if express == 'Micro' and dataset == 'CAS':
            random_num, count = 4, 25000
        elif express == 'Macro' and dataset == 'CAS':
            random_num, count = 5, 40000
        elif express == 'Micro' and dataset == 'SAMM':
            random_num, count = 8, 50000
        else:
            random_num, count = 10, 60000
        info_path = '%s/%s/%s/label/%s_%s.json' % (data_root, dataset, express, uid, mode)
        with open(info_path) as json_file:
            video_info = json.load(json_file)
        json_file.close()

        img_list_true, label_list_true = [], []
        for video_id in video_info:
            annotations = video_info[video_id]['annotations']
            npy_path = '%s/%s/%s/npy/%s.npy' % (data_root, dataset, express, video_id)
            video_data = np.load(npy_path)
            for i in range(video_data.shape[0]):
                img = video_data[i]#cv2.resize(video_data[i], (112, 112))
                label = self.labeling(i, annotations, method=1)
                if mode != 'train':
                    if label > 0:
                        self.label_list.append(label)
                        self.img_list.append(img)
                else:
                    if label > 0:
                        label_list_true.append(label)
                        img_list_true.append(img)
                    elif i % random_num == 0:
                        self.label_list.append(label)
                        self.img_list.append(img)

        if mode == 'train':
            count = min(count, len(self.label_list))
            index = np.random.choice(len(self.label_list), count)
            self.img_list = list(np.array(self.img_list)[index])
            self.label_list = list(np.array(self.label_list)[index])
            if len(label_list_true) > count:
                index = np.random.choice(len(label_list_true), count)
                self.img_list.extend(list(np.array(img_list_true)[index]))
                self.label_list.extend(list(np.array(label_list_true)[index]))
            else:
                num = int(count / len(label_list_true))  # 0.8*
                if num < 1:
                    num = 1
                for j in range(num):
                    for im in img_list_true:
                        self.img_list.append(im)
                        self.label_list.append(1)

        print(mode, 'len: ', len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def labeling(self, im, labels, method=1):
        IOU_list = []
        for label in labels:
            # if im - self.length >= 0:
            #     iou = max(self.cal_IOU([im, im + self.length], label), self.cal_IOU([im - self.length, im], label))
            # else:
            #     iou = self.cal_IOU([im, im + self.length], label)
            # IOU_list.append(iou)
            IOU_list.append(self.cal_IOU([im, (im + self.length)], label))
        if method == 0:
            # iou label 0
            bool_express = False
            for label in labels:
                if im + 1 >= label[0] and im + 1 <= label[1]:
                    bool_express = True
            if bool_express:
                return 1
            else:
                return 0
        elif method == 1:
            # iou label 1
            if max(IOU_list) > 0.5:
                return 1
            else:
                return 0

    def transform_aug(self, image):
        """ Transform a image by cv2.
        """
        # image = cv2.resize(image, (112, 112))  # 128 * 128
        img_size = image.shape[0]
        # horizontal flipping
        if random.random() > 0.8:
            image = cv2.flip(image, 1)
        # # grayscale conversion
        # if random.random() > 0.8:
        #     image = cv2.GaussianBlur(image, (3, 3), 0)
        # if random.random() > 0.8:
        #     image = random_noise(image)
        # grayscale conversion
        # if random.random() > 0.8:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rotation
        if random.random() > 0.8:
            theta = (random.randint(-5, 5)) * np.pi / 180
            M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]],
                                dtype=np.float32)
            image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
        # normalizing
        if image.ndim == 2:
            image = (image - 127.5) * 0.0078125
            new_image = np.zeros([3, img_size, img_size], np.float32)
            new_image[0, :, :] = image
            image = torch.from_numpy(new_image.astype(np.float32))
        else:
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
        return image

    def cal_IOU(self, interval_1, interval_2):
        intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
        if intersection[0] <= intersection[1]:
            len_inter = intersection[1] - intersection[0] + 1
            return len_inter / min(self.length,
                                   interval_2[1] - interval_2[0] + 1)  # len_inter / (interval_1[1] - interval_1[0] + 1)
        else:
            return 0

    def __getitem__(self, index):
        image = self.img_list[index]
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (112, 112))
        image_label = self.label_list[index]
        if self.augment:
            image = self.transform_aug(image)
        else:
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
        return image, image_label

class ImageDatasetImgVal(Dataset):
    def __init__(self, npy_path,  augment=False):
        self.augment = augment
        self.img_list = []

        video_data = np.load(npy_path)
        for i in range(video_data.shape[0]):
            img = cv2.resize(video_data[i], (112, 112))
            self.img_list.append(img)

    def __len__(self):
        return len(self.img_list)


    def transform_aug(self, image):
        """ Transform a image by cv2.
        """
        # image = cv2.resize(image, (112, 112))  # 128 * 128
        img_size = image.shape[0]
        # horizontal flipping
        if random.random() > 0.8:
            image = cv2.flip(image, 1)
        # # grayscale conversion
        # if random.random() > 0.8:
        #     image = cv2.GaussianBlur(image, (3, 3), 0)
        # if random.random() > 0.8:
        #     image = random_noise(image)
        # grayscale conversion
        # if random.random() > 0.8:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rotation
        if random.random() > 0.8:
            theta = (random.randint(-5, 5)) * np.pi / 180
            M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]],
                                dtype=np.float32)
            image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
        # normalizing
        if image.ndim == 2:
            image = (image - 127.5) * 0.0078125
            new_image = np.zeros([3, img_size, img_size], np.float32)
            new_image[0, :, :] = image
            image = torch.from_numpy(new_image.astype(np.float32))
        else:
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
        return image

    def cal_IOU(self, interval_1, interval_2):
        intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
        if intersection[0] <= intersection[1]:
            len_inter = intersection[1] - intersection[0] + 1
            return len_inter / min(self.length,
                                   interval_2[1] - interval_2[0] + 1)  # len_inter / (interval_1[1] - interval_1[0] + 1)
        else:
            return 0

    def __getitem__(self, index):
        image = self.img_list[index]
        if self.augment:
            image = self.transform_aug(image)
        else:
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
        return image

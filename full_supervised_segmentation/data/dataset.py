import sys
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from torch.utils.data import Dataset
from utils.data_augmentation import Compose, ToTensor, CropToFixed, HorizontalFlip, VerticalFlip, RandomRotate90
import cv2
import torch.nn.functional as F
import random

cv2.setNumThreads(0)

VAL_IMAGE_PATH = "/Users/ashmithandoo/Projects/Lab/d_data/validation/images"


class Train_dataset(Dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.size = config.DATASET.PATCH_SIZE
        self.num_each_epoch = config.DATASET.NUM_EACH_EPOCH
        self.image_list = os.listdir(self.images_path)

        seed = np.random.randint(123)

        self.seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

        self.gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

    def __getitem__(self, idx):
        id = random.randint(0, len(self.image_list) - 1)
        img_id = self.image_list[id]
        img =  np.load(os.path.join(self.images_path,img_id))
        gt = cv2.imread(os.path.join(
                self.labels_path, f"label_s{img_id.split('.')[0]}.png"), 0)
        gt = np.array(gt/255)[np.newaxis]
        # gt = np.load(os.path.join(self.labels_path,img_id))[np.newaxis]
        img = self.seq_DA(img)
        gt = self.gt_DA(gt)
        
        return img, gt.long()

    def __len__(self):
        return self.num_each_epoch


class Test_dataset(Train_dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        self.img_list, self.gt_list = self.read_image(
            self.images_path, self.labels_path)
        self.img_patch = self.get_patch(
            self.img_list, self.patch_size, self.stride)
        self.gt_patch = self.get_patch(
            self.gt_list, self.patch_size, self.stride)
    def read_image(self, images_path, label_path):
        label_files = list(sorted(os.listdir(label_path)))
        images = []
        gts = []
        for i in range(len(label_files)):
            if (images_path == VAL_IMAGE_PATH):
                i = i + 30
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    images_path, f"image_s{i}_i{j}.png"), 0)
                if img is not None:
                    image_each_slice.append(img)
                else:
                    image_each_slice.append(np.zeros_like(image_each_slice[0]) if image_each_slice else np.zeros((800, 800), dtype=np.uint8))
            
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)
            images.append(seq)
            
  
            gt = cv2.imread(os.path.join(
                self.labels_path, f"label_s{i}.png"), 0)
            if gt is None:
                ValueError(f"Label file for image_s{i} not found in {self.labels_path}")
            else:
                gt = np.array(gt/255)[np.newaxis]
                # gt = np.load(os.path.join(self.labels_path,f"{i}.npy"))[np.newaxis]
                gts.append(gt)

        return images, gts
    
    def get_patch(self, image_list, patch_size, stride):
        patch_list = []
        _, h, w = image_list[0].shape

        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        for image in image_list:
            image = F.pad(torch.from_numpy(image).float(),
                          (0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(
                2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list

    def __getitem__(self, idx):

        img = self.img_patch[idx]
        gt = self.gt_patch[idx]
        return img, gt.long()

    def __len__(self):
        return len(self.img_patch)

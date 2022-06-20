from torch import argmin
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import albumentations as A
import numpy as np
from glob import glob
from PIL import Image
import torch


def get_training_augmentation(w,h):
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                           shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=h, min_width=w,
                      always_apply=True, border_mode=0),
        A.OneOf(
            [
                A.RandomCrop(height=h, width=w, always_apply=True),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST),
            ],
            p=1,
        ),


        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.6,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.6,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(w, h):
    """Add paddings to make image shape divisible by 32"""
    return A.Compose([A.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST)])


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (Amentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (Amentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_list,
            masks_list,
            preprocessing=None,
            augmentation=None,
    ):
        self.ids = images_list
        self.images_fps = images_list
        self.masks_fps = masks_list

        # convert str names to class values on masks

        # self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. cars)
        # apply preprocessing
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            image = Image.fromarray(image)

        else:
            image = Image.fromarray(image)

        if self.preprocessing:
            image = self.preprocessing(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.ids)


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, drop_last=True)

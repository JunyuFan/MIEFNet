import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from .transform import *
import matplotlib.patches as mpatches
from PIL import Image
import random


CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)

def get_training_transform():
    train_transform = [
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask, dsm, ir):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask, dsm, ir = crop_aug(img, mask, dsm, ir)
    img, mask, dsm, ir = np.array(img), np.array(mask), np.array(dsm), np.array(ir)
    masks = [mask, dsm, ir]
    aug = get_training_transform()(image=img.copy(), masks=masks.copy())
    img, masks = aug['image'], aug['masks']
    mask, dsm, ir = masks[0], masks[1], masks[2]
    return img, mask, dsm, ir


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask, dsm, ir):
    img, mask, dsm, ir = np.array(img), np.array(mask), np.array(dsm), np.array(ir)
    masks = [mask, dsm, ir]
    aug = get_val_transform()(image=img.copy(), masks=masks.copy())
    img, masks = aug['image'], aug['masks']
    mask, dsm, ir = masks[0], masks[1], masks[2]
    return img, mask, dsm, ir


class PotsdamDataset(Dataset):
    def __init__(self, data_root='data/potsdam/test', mode='val', img_dir='images', mask_dir='masks', dsm_dir='dsms', ir_dir='irs',
                 img_suffix='.tif', mask_suffix='.png', dsm_suffix='.tif', ir_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dsm_dir = dsm_dir
        self.ir_dir = ir_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dsm_suffix = dsm_suffix
        self.ir_suffix = ir_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir, self.dsm_dir, self.ir_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask, dsm, ir = self.load_img_and_mask(index)
            if self.transform:
                img, mask, dsm, ir = self.transform(img, mask, dsm, ir)
        else:
            img, mask, dsm, ir = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask, dsm, ir = self.transform(img, mask, dsm, ir)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        ir = torch.from_numpy(ir).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        dsm = torch.from_numpy(dsm).unsqueeze(0)
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask, dsm=dsm, ir=ir)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir, dsm_dir, ir_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        dsm_filename_list = os.listdir(osp.join(data_root, dsm_dir))
        ir_filename_list = os.listdir(osp.join(data_root, ir_dir))
        assert len(img_filename_list) == len(mask_filename_list) and len(img_filename_list) == len(dsm_filename_list) and len(img_filename_list) == len(ir_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        dsm_name = osp.join(self.data_root, self.dsm_dir, img_id + self.dsm_suffix)
        ir_name = osp.join(self.data_root, self.ir_dir, img_id + self.ir_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        dsm = Image.open(dsm_name)
        ir = Image.open(ir_name)
        return img, mask, dsm, ir    #1024, 1024

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a, dsm_a, ir_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b, dsm_b, ir_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c, dsm_c, ir_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d, dsm_d, ir_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a, dsm_a, ir_a = np.array(img_a), np.array(mask_a), np.array(dsm_a), np.array(ir_a)
        img_b, mask_b, dsm_b, ir_b = np.array(img_b), np.array(mask_b), np.array(dsm_b), np.array(ir_b)
        img_c, mask_c, dsm_c, ir_c = np.array(img_c), np.array(mask_c), np.array(dsm_c), np.array(ir_c)
        img_d, mask_d, dsm_d, ir_d = np.array(img_d), np.array(mask_d), np.array(dsm_d), np.array(ir_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        masks_a = [mask_a, dsm_a, ir_a]
        masks_b = [mask_b, dsm_b, ir_b]
        masks_c = [mask_c, dsm_c, ir_c]
        masks_d = [mask_d, dsm_d, ir_d]

        croped_a = random_crop_a(image=img_a.copy(), masks=masks_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), masks=masks_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), masks=masks_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), masks=masks_d.copy())

        img_crop_a, masks_crop_a = croped_a['image'], croped_a['masks']
        img_crop_b, masks_crop_b = croped_b['image'], croped_b['masks']
        img_crop_c, masks_crop_c = croped_c['image'], croped_c['masks']
        img_crop_d, masks_crop_d = croped_d['image'], croped_d['masks']

        mask_crop_a, dsm_crop_a, ir_crop_a = masks_crop_a[0], masks_crop_a[1], masks_crop_a[2]
        mask_crop_b, dsm_crop_b, ir_crop_b = masks_crop_b[0], masks_crop_b[1], masks_crop_b[2]
        mask_crop_c, dsm_crop_c, ir_crop_c = masks_crop_c[0], masks_crop_c[1], masks_crop_c[2]
        mask_crop_d, dsm_crop_d, ir_crop_d = masks_crop_d[0], masks_crop_d[1], masks_crop_d[2]

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        top_dsm = np.concatenate((dsm_crop_a, dsm_crop_b), axis=1)
        bottom_dsm = np.concatenate((dsm_crop_c, dsm_crop_d), axis=1)
        dsm = np.concatenate((top_dsm, bottom_dsm), axis=0)

        top_ir = np.concatenate((ir_crop_a, ir_crop_b), axis=1)
        bottom_ir = np.concatenate((ir_crop_c, ir_crop_d), axis=1)
        ir = np.concatenate((top_ir, bottom_ir), axis=0)


        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        dsm = np.ascontiguousarray(dsm)
        ir = np.ascontiguousarray(ir)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        dsm = Image.fromarray(dsm)
        ir = Image.fromarray(ir)
        return img, mask, dsm, ir

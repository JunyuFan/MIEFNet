import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os
from skimage import io


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def building_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def pv2rgb(mask):  # Potsdam and vaihingen
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=Path, default=r'data/potsdam/test_images', help="Path to  huge image folder")
    arg("-dsm", "--dsm_path", type=Path, default=r'data/potsdam/test_dsms', help="Path to  huge dsm folder")
    arg("-c", "--config_path", type=Path, default=r'config/potsdam/config.py', help="Path to config")
    arg("-o", "--output_path", type=Path, default=r'fig_results/potsdam/Ours_huge', help="Path where to save resulting masks.")
    arg("-t", "--tta", help="Test time augmentation.", default='d4', choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=1024)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=1024)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    arg("-d", "--dataset", help="dataset", default="pv", choices=["pv", "landcoverai", "uavid", "building"])
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                           border_mode=0, value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, img_tile_list=None, dsm_tile_list=None, ir_tile_list=None, transform=albu.Normalize()):
        self.img_tile_list = img_tile_list
        self.dsm_tile_list = dsm_tile_list
        self.ir_tile_list = ir_tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.img_tile_list[index]
        dsm = self.dsm_tile_list[index]
        ir = self.ir_tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        ir = torch.from_numpy(ir).permute(2, 0, 1).float()
        dsm = torch.from_numpy(dsm).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img, dsm=dsm, ir=ir)
        return results

    def __len__(self):
        return len(self.img_tile_list)


def make_dataset_for_one_huge_image(img_path, dsm_path, patch_size):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = io.imread(img_path)
    ir = img[:, :, 3]       #vaihingen=2 potsdam=3
    ir = np.expand_dims(ir, axis=2)
    img = img[:, :, :3]
    dsm = Image.open(dsm_path)
    dsm = np.array(dsm)
    dsm = np.expand_dims(dsm, axis=2)
    img_tile_list = []
    dsm_tile_list = []
    ir_tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)
    dsm_pad, _, _ = get_img_padded(dsm.copy(), patch_size)
    ir_pad, _, _ = get_img_padded(ir.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            dsm_tile = dsm_pad[x:x+patch_size[0], y:y+patch_size[1]]
            ir_tile = ir_pad[x:x+patch_size[0], y:y+patch_size[1]]
            img_tile_list.append(image_tile)
            dsm_tile_list.append(dsm_tile)
            ir_tile_list.append(ir_tile)

    dataset = InferenceDataset(img_tile_list=img_tile_list, dsm_tile_list=dsm_tile_list, ir_tile_list=ir_tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


def main():
    args = get_args()
    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)

    model.cuda(config.gpus[0])
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        # model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
            ]
        )
        # model = tta.SegmentationTTAWrapper(model, transforms)

    img_paths = []
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for ext in ('*.tif', '*.png', '*.jpg'):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()
    # print(img_paths)
    for img_path in img_paths:
        img_name = img_path.split('\\')[-1]
        dsm_path = os.path.join(args.dsm_path, img_name)
        # print('origin mask', original_mask.shape)
        dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
            make_dataset_for_one_huge_image(img_path, dsm_path, patch_size)
        # print('img_padded', img_pad.shape)
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_tiles = []
        k = 0
        with torch.no_grad():
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                    drop_last=False, shuffle=False)
            for input in tqdm(dataloader):

                if args.tta in ['d4', 'lr']:
                    pred = None
                    for transformer in transforms:
                        vis = transformer.augment_image(input['img'].cuda(config.gpus[0]))
                        ir = transformer.augment_image(input['ir'].cuda(config.gpus[0]).float())
                        dsm = transformer.augment_image(input['dsm'].cuda(config.gpus[0]))

                        aug_pred = model(vis, ir, dsm)
                        deaug_pred = transformer.deaugment_mask(aug_pred)
                        if pred is None:
                            pred = deaug_pred
                        pred += deaug_pred
                    raw_predictions = pred / len(transforms)
                else:
                    raw_predictions = model(input['img'].cuda(config.gpus[0]), input['ir'].cuda(config.gpus[0]), input['dsm'].cuda(config.gpus[0]))
                    

                raw_predictions = nn.Softmax(dim=1)(raw_predictions)

                predictions = raw_predictions.argmax(dim=1)
                image_ids = input['img_id']


                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    output_tiles.append((mask, image_ids[i].cpu().numpy()))

        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                # print(output_tiles[k][1])
                k = k + 1

        output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]

        if args.dataset == 'pv':
            output_mask = pv2rgb(output_mask)
        else:
            output_mask = output_mask

        cv2.imwrite(os.path.join(args.output_path, img_name), output_mask)


if __name__ == "__main__":
    main()

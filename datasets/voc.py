"""
dataset： PASCAL VOC

"""
import os
import cv2
import torch
import pickle
import random
import subprocess
import numpy as np
import os.path as osp
import numpy.random as npr
import scipy.io as sio
import scipy.sparse
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as eTree
import torchvision.transforms as transforms
from datasets.utils import normalize_image, get_im_scale


NUM_CLASSES = 21

CLASSES = ('__background__',
           'aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor')


class VOC2012(Dataset):

    def __init__(self, dataroot, imageset, config):
        assert imageset == 'train' or imageset == 'val' or imageset == 'trainval' or imageset == 'test'
        self.imageset = imageset
        image_name_txt = osp.join(dataroot, 'VOC2012', 'ImageSets', 'Main', '{}.txt'.format(imageset))
        self.image_dir = osp.join(dataroot, 'VOC2012', 'JPEGImages')

        self.annotation_dir = osp.join(dataroot, 'VOC2012', 'Annotations')
        with open(image_name_txt, 'r') as f:
            image_names = f.readlines()
        self.image_names = [x.rstrip('\n') for x in image_names]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        self.image_size = 600  # (480, 576, 688, 864, 1200)
        self.cls2idx = dict(zip(CLASSES, range(NUM_CLASSES)))
        self.config = config

    def __len__(self):
        return len(self.image_names)

    def _load_annotation_from(self, im_name):
        annotation_path = os.path.join(self.annotation_dir, im_name+'.xml')
        tree = eTree.parse(annotation_path)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')

            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls_name = obj.find('name').text.lower().strip()
            cls = self.cls2idx[cls_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return gt_classes, boxes

    def __getitem__(self, idx):
        im_name = self.image_names[idx]
        img = Image.open(os.path.join(self.image_dir, im_name+'.jpg'))

        if self.imageset == 'test' or self.imageset == 'val':
            # testing or validation mode, original scale
            img = np.array(img).astype('float32')
            h, w = img.shape[:2]
            resize_h, resize_w, scale = get_im_scale(h, w, target_size=self.config['test_image_size'][0],
                                                     max_size=self.config['test_max_image_size'])
            img = cv2.resize(img, (resize_w, resize_h))
            img = normalize_image(img)
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img)
            return img, im_name, scale, (h, w)

        img = np.array(img).astype('float32')
        labels, boxes = self._load_annotation_from(im_name)
        labels = torch.LongTensor(labels)
        # C, H, W

        return img, labels, boxes

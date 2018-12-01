"""

"""
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import pycocotools.coco as COCO
from datasets.utils import normalize_image, get_im_scale
from torch.utils.data import Dataset
from torchvision.transforms import transforms

CLASSES = (
    'background', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')


class COCODetection(Dataset):
    """ COCO Detection Dataset

    dataroot [annotations, train2017, val2017]
    imageset [train2017, val2017]
    """
    def __init__(self, dataroot, config, imageset='train2017'):
        assert imageset == 'train2017' or 'val2017'
        # train2017/ val2017
        self.imageset = imageset
        self.config = config
        self.images_dir = os.path.join(dataroot, imageset)
        annotation_path = os.path.join(dataroot, 'annotations', 'instances_{}.json'.format(imageset))
        self.coco_helper = COCO.COCO(annotation_path)
        self.img_ids = list(self.coco_helper.imgs.keys())

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        catids = self.coco_helper.getCatIds()
        catids = [0] + catids
        self.catid2id = dict(zip(catids, range(len(catids))))
        self.id2catid = dict(zip(range(len(catids)), catids))

    def _get_category_map(self):
        cats = self.coco_helper.loadCats(self.coco_helper.getCatIds())
        cats = [(x['name'], x['id']) for x in cats]
        cat_id_map = dict(cats)
        return cat_id_map

    def load_annotation(self, img_id):
        annotation = self.coco_helper.loadAnns(self.coco_helper.getAnnIds(img_id))
        boxes = [(ann['bbox'], self.catid2id[ann['category_id']]) for ann in annotation]
        return boxes

    def get_img_path(self, img_id):
        info = self.coco_helper.imgs[img_id]
        filename = info['file_name']
        filepath = os.path.join(self.images_dir, filename)
        return filepath

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        im_id = self.img_ids[idx]
        img_path = self.get_img_path(im_id)
        img = Image.open(img_path).convert('RGB')

        if self.imageset == 'val2017':
            img = np.array(img).astype('float32')
            h, w = img.shape[:2]
            resize_h, resize_w, scale = get_im_scale(h, w, target_size=self.config['test_image_size'][0],
                                                     max_size=self.config['test_max_image_size'])
            img = cv2.resize(img, (resize_w, resize_h))
            img = normalize_image(img)
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img)
            return img, im_id, scale, (h, w)

        annotations = self.load_annotation(im_id)
        boxes = np.array([x[0] for x in annotations], dtype='float32')
        if boxes.shape[0] == 0:
            boxes = np.array([[0, 0, 0, 0]], dtype='float32')
        # x1,y1,w,h -> x1,y1,x2,y2
        boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2] - 1
        # boxes = np_xywh2xyxy(boxes)
        labels = torch.LongTensor([x[1] for x in annotations])
        img = np.array(img).astype('float32')

        # labels N
        # onehot_labels = torch.zeros((labels.shape[0], config.num_classes))
        # onehot_labels[range(labels.shape[0]), labels] = 1
        # B*N
        return img, labels, boxes

"""

Convert .det result file to VOC2012 format

"""

import os
import json
import argparse
import sys
sys.path.append('../')
from datasets.voc import CLASSES


def convert(detpath, vocdir):

    with open(detpath, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\n'), lines))

    assert os.path.exists(vocdir)

    for line in lines:
        res = json.loads(line)
        im_name = res['image_id']
        bboxes = res['result']
        for bbox in bboxes:
            prob = bbox['prob']
            box = bbox['bbox']
            cls = bbox['class']
            class_name = CLASSES[cls]
            with open('{}/comp4_det_val_{}.txt'.format(vocdir, class_name), 'a+') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    im_name, prob, box[0], box[1], box[2], box[3]
                ))
    print("convert to {} finsihed".format(vocdir))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detpath', type=str, help='.det file path')
    parser.add_argument('-v', '--vocdir', type=str, help='voc file directory')

    args = parser.parse_args()
    convert(args.detpath, args.vocdir)


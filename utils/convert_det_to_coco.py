import os
import json
import argparse
import sys
sys.path.append('../')
# from datasets import coco
from pycocotools import coco
def convert(detpath, jsonpath):
    co = coco.COCO('/hdfs/resrchvc/v-tich/cls/data/coco/annotations/instances_val2017.json')
    ids = co.getCatIds()   
    with open(detpath, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\n'), lines))
    result = []
    for line in lines:
        res = json.loads(line)
        im_name = res['image_id']
        bboxes = res['result']
        for bbox in bboxes:
            prob = bbox['prob']
            box = bbox['bbox']
            cls = ids[bbox['class']-1]
            result.append(
                {
                 'image_id': im_name,
                 'category_id': cls,
                 'bbox': [box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1],
                 'score': prob
                }
            )

    json_str = json.dumps(result)
    with open(jsonpath, 'w') as f:
        f.write(json_str)



if __name__ == '__main__':
    convert('../result.det', '../result.json')






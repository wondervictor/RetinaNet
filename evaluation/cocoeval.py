"""

"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pycocotools.coco as COCO
import pycocotools.cocoeval as COCOeval


def coco_bbox_eval(result_file, annotation_file):

    ann_type = 'bbox'
    coco_gt = COCO.COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)
    cocoevaler = COCOeval.COCOeval(coco_gt, coco_dt, ann_type)
    cocoevaler.evaluate()
    cocoevaler.accumulate()
    cocoevaler.summarize()


def coco_proposal_eval(result_file, annotation_file):

    ann_type = 'bbox'
    coco_gt = COCO.COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)
    cocoevaler = COCOeval.COCOeval(coco_gt, coco_dt, ann_type)
    cocoevaler.params.useCats = 0
    cocoevaler.evaluate()
    cocoevaler.accumulate()
    cocoevaler.summarize()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result', type=str, default='', help='detection result file')
    parser.add_argument('-a', '--annotation', type=str,
                        default='/public_datasets/COCO/annotations/instances_val2017.json',
                        help='COCO groundtruth')
    parser.add_argument('-t', '--type', type=str, default='bbox', help='eval type: [bbox, seg, proposal]')
    _args = parser.parse_args()

    if _args.type == 'bbox':
        coco_bbox_eval(_args.result, _args.annotation)
    elif _args.type == 'proposal':
        coco_proposal_eval(_args.result, _args.annotation)
    else:
        pass



import torch, torchvision
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import json
# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file: json.dump(data, file)


def main():
    # prepare predictor
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    category_name_list = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    print("name list")
    print(category_name_list)
    print()
    det_id = 0
    dets = []
    cnt = 0
    dataa =[]

    # process one image
    image_id = 0
    #while image_id < 8349:
    while image_id < 3:
        im = cv2.imread("./data/talk2car/imgs/img_train_{}.jpg".format(image_id))
        outputs = predictor(im)

        # check if no bounding box
        num_boxes = len(outputs["instances"].pred_boxes)
        #if num_boxes == 0:
            #raise ValueError
        
        # count 
        scores = outputs["instances"].scores.tolist()
        print("scores: ", scores)
        threshold = 0.65
        num_valid_boxes = sum( [s>=threshold for s in scores] )
        print("num valid boxes: ", num_valid_boxes)

        while num_valid_boxes == 0:
            threshold -= 0.1
            num_valid_boxes = sum( [s>=threshold for s in scores] )
            if threshold < 0.35:
                raise ValueError

        for i in range(num_boxes):
            score = scores[i]
            if score < threshold:
                continue

            box_i = outputs["instances"].pred_boxes[i].tensor
            x1, y1, x2, y2 = box_i[0, 0].item(), box_i[0, 1].item(), box_i[0, 2].item(), box_i[0, 3].item()
            
            category_id = outputs["instances"].pred_classes[i].item()
            category_name = category_name_list[category_id]

            det = {'det_id': det_id,
                   'h5_id': det_id,  # we make h5_id == det_id
                   'box': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                   'image_id': image_id,
                   'category_id': category_id + 1,  # because they skip background
                   'category_name': category_name,
                   'score': score}
            dets += [det]
            det_id+=1
        image_id += 1
    
    out_json_path = "./talk2car_dets.json"
    with open(out_json_path, 'w') as f:
        json.dump(dets, f)


if __name__ == "__main__":
    main()

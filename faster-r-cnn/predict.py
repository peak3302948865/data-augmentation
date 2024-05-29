import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model.faster_RCNN import create_model
import random
import numpy as np
from loguru import logger
from pathlib import Path
from configs.fasterrcnn_confs import get_cfg_defaults
from data.dataset import VocCustomDataset, create_train_loader, create_valid_loader
from utils.data_utils import get_train_transform,get_valid_transform,draw_box
import torchvision
from tqdm import tqdm
import argparse

def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./configs/train_confs.yaml")
    cfg.freeze()
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    # make output dir
    logger.add(cfg.LOG_PATH,
        level='DEBUG',
        format='{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}',
        rotation="10 MB")
    logger.info("Train config: %s" % str(cfg))
    model_output = Path(cfg.MODEL.saved_path)
    model_output.mkdir(exist_ok =True)
    return cfg

def tensor2numpy(input_tensor):
	input_tensor=input_tensor.to(torch.device('cpu')).numpy()
	in_arr=np.transpose(input_tensor,(1,2,0))	# transform (c,w,h) to (w,h,c), then to cv2 objective
	return cv2.cvtColor(np.uint8(in_arr*255), cv2.COLOR_BGR2RGB)



def main(cfg, dv, df):
    CLASSES = cfg.DATA.classes 
    NUM_CLASSES = cfg.DATA.num_classes
    DEVICE = cfg.TRAIN.device

    draw_VOC = dv
    draw_firststage = df

    model_path = Path(cfg.MODEL.saved_path)/'best_model.pt'
    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()

    # directory where all the images are present
    test_dataset = VocCustomDataset(cfg,'test',get_valid_transform())
    
    test_loader = create_valid_loader(cfg,test_dataset)

    # define the detection threshold
    detection_threshold = 0.8

    # to count the total number of images iterated through
    frame_count = 0
    # to keep adding the FPS for each image
    total_fps = 0

    if draw_VOC:
        for i,data in enumerate(tqdm(test_loader)):
            # get the image file name for saving output later on
            img,label = data
            img = [m.to(DEVICE) for m in img]
            test_image = img[0]
            test_label = label[0]
            
            orig_image = tensor2numpy(test_image)
            orig_image = cv2.resize(orig_image, test_label['origin_shape'])
            
            # make the pixel range between 0 and 1
            start_time = time.time()
            with torch.no_grad():
                outputs = model(img)
            end_time = time.time()

            # get the current fps
            fps = 1 / (end_time - start_time)
            # add `fps` to `total_fps`
            total_fps += fps
            # increment frame count
            frame_count += 1
            
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
#            print(outputs[0])
            # carry further only if there are detected boxes
            if not draw_firststage:
                if len(outputs[0]['boxes']) != 0:
                    boxes = outputs[0]['boxes'].data.numpy()
                    width,height = test_image.shape[1],test_image.shape[2]
                    boxes[:,0] = (boxes[:,0]/width)*test_label['origin_shape'][0]
                    boxes[:,2] = (boxes[:,2]/width)*test_label['origin_shape'][0]
                    boxes[:,1] = (boxes[:,1]/height)*test_label['origin_shape'][1]
                    boxes[:,3] = (boxes[:,3]/height)*test_label['origin_shape'][1]

                    scores = outputs[0]['scores'].data.numpy()
                    # filter out boxes according to `detection_threshold`
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    draw_boxes = boxes.copy()
                    # get all the predicited class names
                    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
                    
                    # draw the bounding boxes
                    draw_box(orig_image,draw_boxes,pred_classes,scores,CLASSES,COLORS,label=True)
#                    cv2.waitKey(1)
                    cv2.imwrite(f"imgs/output/final_result/{test_label['image_name']}.jpg", orig_image)
                    if i > 2:
                        break
            else:
               scores = outputs[0]['scores']
               sorted_score, indices = torch.sort(scores, descending=True) 
               boxes = outputs[0]['boxes'][indices][:50].data.numpy()
               
               
               width,height = test_image.shape[1],test_image.shape[2]
               boxes[:,0] = (boxes[:,0]/width)*test_label['origin_shape'][0]
               boxes[:,2] = (boxes[:,2]/width)*test_label['origin_shape'][0]
               boxes[:,1] = (boxes[:,1]/height)*test_label['origin_shape'][1]
               boxes[:,3] = (boxes[:,3]/height)*test_label['origin_shape'][1]
               draw_boxes = boxes.copy()
           
               # draw the bounding boxes
               draw_box(orig_image,draw_boxes,None,None,CLASSES,COLORS,label=False)
#               cv2.waitKey(1)
               cv2.imwrite(f"imgs/output/first_stage/{test_label['image_name']}.jpg", orig_image)
               if i > 2:
                   break
               
            print(f"Image {i+1} done...")
            print('-'*50)

        print('TEST PREDICTIONS COMPLETE')

    else:
        # draw object detection results for figure NOT in VOC2007
        detection_threshold = 0.7
        image_dir = [p for p in Path('imgs/input').rglob("*.jpg")]
        for img_path in image_dir:
            image = cv2.imread(str(img_path))
            orig_image = image.copy()
            # BGR to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # make the pixel range between 0 and 1
            image /= 255.0
            image = cv2.resize(image, (416, 416))
            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            # convert to tensor
            image = torch.tensor(image, dtype=torch.float).to(DEVICE)
            # add batch dimension
            
            image = torch.unsqueeze(image, 0)
            image_shape = np.array(np.shape(image[0])[1:])
            with torch.no_grad():
                outputs = model(image.to(DEVICE))
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                width,height = image_shape
                boxes[:,0] = (boxes[:,0]/width)*orig_image.shape[1]
                boxes[:,2] = (boxes[:,2]/width)*orig_image.shape[1]
                boxes[:,1] = (boxes[:,1]/height)*orig_image.shape[0]
                boxes[:,3] = (boxes[:,3]/height)*orig_image.shape[0]

                scores = outputs[0]['scores'].data.numpy()
                # filter out boxes according to `detection_threshold`
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicited class names
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

                # draw the bounding boxes
                draw_box(orig_image,draw_boxes,pred_classes,scores,CLASSES,COLORS,label=True)
#                cv2.waitKey(1)
                cv2.imwrite(f"imgs/output/other_pic/{img_path.name}", orig_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw_VOC', '-dv', action='store_true', default=False)
    parser.add_argument('--draw_firststage', '-df', action='store_true', default=False)
    args = parser.parse_args()
    
    cfg = prepare_config()
    main(cfg, dv=args.draw_VOC, df=args.draw_firststage)
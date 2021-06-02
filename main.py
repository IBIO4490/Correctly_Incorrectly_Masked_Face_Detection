import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import glob
import argparse
import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import urllib
import json
import PIL.Image as Image
import scipy.io

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

from IPython.display import Image, display 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import load_model
import PIL
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import torch

# Arguments
parser = argparse.ArgumentParser(description='Face mask detection, the algorithm requires detectron2' )
parser.add_argument('--mode', type=str, default='demo', metavar='N',  
                    help='Selects the type of prove the algorithm perfomance')
parser.add_argument('--img', type=str, default='test_00000006.jpg', metavar='N',  
                    help='Selects the image in which the algorith perform')

args = parser.parse_args()

dataroot= '/home/saotero/ProyectoFinal'

modelo=load_model(os.path.join(dataroot ,'classifier.h5'))



def demo_classifier(rgb_image, modelo):
    clases=['Mask','Mask_Chin','Mask_Mouth_Chin','Mask_Nose_Mouth']
    temp=tf.image.convert_image_dtype(rgb_image, dtype=tf.float16, saturate=False, name=None)
    temp=tf.expand_dims(temp,axis=0)
    pr=modelo.predict(temp)
    index=np.argmax(pr[0])
    prediction=clases[index]
    print('The predicted label is: '+str(prediction))
    return prediction
  

torch.cuda.set_device(2)
print('GPU: ',torch.cuda.current_device())



df = pd.read_csv(os.path.join(dataroot ,'annotationsC5.csv'))
dfT = pd.read_csv(os.path.join(dataroot ,'annotationsC5T.csv'))

IMAGES_PATH = os.path.join(dataroot ,'train-images','images')
IMAGES_PATHT=os.path.join(dataroot ,'test-images','images')
unique_files = df.file_name.unique()
unique_filesT = dfT.file_name.unique()

train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.95), replace=False))
test_files = set(np.random.choice(unique_filesT, int(len(unique_filesT) * 1), replace=False))
train_df = df[df.file_name.isin(train_files)]
val_df = df[~df.file_name.isin(train_files)]
test_df = dfT[dfT.file_name.isin(test_files)]
classes = df.class_name.unique().tolist()

def create_dataset_dicts(df, classes,IMAGES_PATH):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):
        record = {}

        image_df = df[df.file_name == img_name]

        file_path =  os.path.join(IMAGES_PATH,img_name)
        record["file_name"] = file_path
        record["image_id"] = image_id 
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)

        objs = []
        
        for _, row in image_df.iterrows():
            xmin = int(row.x)
            ymin = int(row.y)
            xmax = int(row.w)
            ymax = int(row.h)

            poly = [
              (xmin, ymin), (xmax, ymin),
              (xmax, ymax), (xmin, ymax)
          ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": classes.index(row.class_name),
            "iscrowd": 0
          }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val","test"]:
  if d == "test":
    DatasetCatalog.register("faces_" + d, lambda d=d: create_dataset_dicts(test_df, classes,IMAGES_PATHT))
    MetadataCatalog.get("faces_" + d).set(thing_classes=classes)
  else:
    DatasetCatalog.register("faces_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else val_df, classes,IMAGES_PATH))
    MetadataCatalog.get("faces_" + d).set(thing_classes=classes)

statement_metadata = MetadataCatalog.get("faces_test")

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_evalRXF", exist_ok=True)
        output_folder = "coco_evalRXF"
    print(dataset_name)

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()

cfg.merge_from_file(
  model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
  )
)


cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
  "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)


cfg.DATASETS.TRAIN = ("faces_train",)
# cfg.DATASETS.VAL = ("faces_val",)
cfg.DATASETS.TEST = ("faces_test",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (500, 1500)
cfg.SOLVER.GAMMA = 0.02

print(classes)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

cfg.TEST.EVAL_PERIOD = 5000
cfg.OUTPUT_DIR='outputRXF'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


trainer = CocoTrainer(cfg)
trainer.resume_or_load()
# print('EVALUACIONPREVIA:-----------------------------------------------------------------------------------')
# evaluator = COCOEvaluator("faces_test", cfg, False, cfg.OUTPUT_DIR)
# val_loader = build_detection_test_loader(cfg, "faces_test")
# inference_on_dataset(trainer.model, val_loader, evaluator)



# trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
if args.mode == 'test':
  print('EVALUACION:-----------------------------------------------------------------------------------')
  
  val_loader = build_detection_test_loader(cfg, "faces_test")
  inference_on_dataset(trainer.model, val_loader, evaluator)
  print('RESULTADOS:-----------------------------------------------------------------------------------')
else:
  test_image_paths = args.img

  file_path =  os.path.join(IMAGES_PATHT,test_image_paths)
  im = Image.open(file_path)
  imT = cv2.imread(file_path)
  outputs = predictor(imT)
  instances = outputs["instances"].to("cpu")
  Boxes=outputs['instances'].pred_boxes
  for i in Boxes:
    variable=i.cpu().numpy()
    im1=im.crop(variable)
    im1=im1.resize(128,128)
    label=demo_classifier(im1,modelo)
    color=(0, 255, 0)
    color1=(255,0,0)
    color2=(255, 255, 255)
    color3=(0,0,255)
    if label =='Mask':
      cv2.rectangle(im1, variable, color, 2)
      cv2.putText(im1, label, (variable[0], variable[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    elif label == 'Mask_Chin':
      cv2.rectangle(im1, variable, color1, 2)
      cv2.putText(im1, label, (variable[0], variable[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, 2)
    elif label == 'Mask_Mouth_Chin':
      cv2.rectangle(im1, variable, color2, 2)
      cv2.putText(im1, label, (variable[0], variable[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color2, 2)
    else:
      cv2.rectangle(im1, variable, color3, 2)
      cv2.putText(im1, label, (variable[0], variable[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color3, 2)
      im1.save(os.path.join('RESULT',test_image_paths))
    




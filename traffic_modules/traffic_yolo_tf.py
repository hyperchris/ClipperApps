import os 
import sys 
import numpy as np
from time import time 
from os.path import join 

# Import darkflow to sys patrh 
df_home = join(os.getcwd(), 'darkflow')
if os.path.exists(df_home):
    sys.path.insert(0, df_home)
else:
    print("Warning: darkflow is not in current dir!!!")

# Darkflow should be installed from: https://github.com/thtrieu/darkflow
from darkflow.net.build import TFNet
# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = 'checkpoints/yolo/cfg'
YOLO_MODEL = 'checkpoints/yolo/cfg/yolo.cfg'
YOLO_WEIGHTS = 'checkpoints/yolo/bin/yolo.weights'

GPU_ID = 0
GPU_UTIL = 0.5
YOLO_THRES = 0.2
YOLO_PEOPLE_LABEL = 'person'


import base64
import cv2
import json
import requests


class YOLO:
    dets = []
    input = {}

    def Setup(self):
        opt = { "config": YOLO_CONFIG,  
                "model": YOLO_MODEL, 
                "load": YOLO_WEIGHTS, 
                "gpuName": GPU_ID,
                "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        self.tfnet = TFNet(opt)

    def PreProcess(self, input):
        self.input = input 

    def Apply(self):
        self.dets = self.tfnet.return_predict(self.input['img'])   

    def PostProcess(self):
        output = self.input
        output['meta']['obj'] = []
        
        for d in self.dets:
            if d['label'] != YOLO_PEOPLE_LABEL:
                continue 
            output['meta']['obj'].append({
                'box':[
                    int(d['topleft']['x']), 
                    int(d['topleft']['y']), 
                    int(d['bottomright']['x']), 
                    int(d['bottomright']['y'])
                ],
                'label': d['label'],
                'conf': d['confidence'],
            })

        return output

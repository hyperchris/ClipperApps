import os 
import sys 
import numpy as np 
from time import time 
from os.path import join 


# Download the model file to 'checkpoints/'
DEEPSORT_MODEL = join(os.getcwd(),'checkpoints/deepsort/mars-small128.pb')

DS_HOME = join(os.getcwd(), 'tracking_modules/deep_sort')
sys.path.insert(0, DS_HOME)
# The original DS tools folder doesn't have init file, add it
fout = open(join(DS_HOME, 'tools/__init__.py'), 'w')
fout.close()
from tools.generate_detections import create_box_encoder

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }
'''

import base64
import cv2 
import json
import requests

def clipper_query(addr, img):
    url = "http://%s/track2-endpoint/predict" % addr
    retval, buf = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buf)
    req_json = json.dumps({
        "input":
        jpg_as_text.decode() # bytes to unicode
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    
class FeatureExtractor:
    ds_boxes = []
    input = {}
    features = []

    def Setup(self):
        self.encoder = create_box_encoder(DEEPSORT_MODEL, batch_size=16)
        self.log('init')

    def PreProcess(self, input):
        assert input, "REID: empty input!"
        self.input = input

        boxes = input['meta']['obj']
        self.ds_boxes = [[b['box'][0], b['box'][1], b['box'][2] - b['box'][0], 
                                    b['box'][3] - b['box'][1]] for b in boxes]

    def Apply(self):
        ''' Extract features and update the tracker 
        ''' 
        self.features = self.encoder(self.input['img'], self.ds_boxes)

    def PostProcess(self):
        output = self.input
        for i in range(len(self.ds_boxes)):
            output['meta']['obj'][i]['box'] = self.ds_boxes[i]
            output['meta']['obj'][i]['feature'] = self.features[i]
            
        return output 

    def log(self, s):
        print('[FExtractor] %s' % s)


import os 
import sys 
import math
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
import base64
from clipper_start import register, predict 

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = 'checkpoints/yolo/cfg'
YOLO_MODEL = 'checkpoints/yolo/cfg/yolo.cfg'
YOLO_WEIGHTS = 'checkpoints/yolo/bin/yolo.weights'

GPU_ID = 0
GPU_UTIL = 0.4
YOLO_THRES = 0.2
YOLO_PEOPLE_LABEL = 'person'


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
        self.clipper_model_name = 'traffic-yolo'
        self.sess = self.tfnet.sess
        self.clipper_conn = register(
          model_name=self.clipper_model_name, 
          sess=self.sess, 
          func=self.run_session,
        )
        print('YOLO READY')

    def PreProcess(self, input):
        self.input = input 

    def run_session(self, imgs):
        im = cv2.imdecode(np.fromstring(base64.b64decode(imgs[0]), dtype=np.uint8), 1)
        h, w, _ = im.shape
        im = self.tfnet.framework.resize_input(im)
        this_inp = np.expand_dims(im, 0)
        feed_dict = {self.tfnet.inp : this_inp}

        out = self.sess.run(self.out, feed_dict)[0]
        boxes = self.tfnet.framework.findboxes(out)
        threshold = self.tfnet.FLAGS.threshold
        boxesInfo = list()
        for box in boxes:
            tmpBox = self.tfnet.framework.process_box(box, h, w, threshold)
            if tmpBox is None:
                continue
            boxesInfo.append({
                "label": tmpBox[4],
                "confidence": tmpBox[6],
                "topleft": {
                    "x": tmpBox[0],
                    "y": tmpBox[2]},
                "bottomright": {
                    "x": tmpBox[1],
                    "y": tmpBox[3]}
            })
        return boxesInfo

    def Apply(self):
        self.dets = predict(
            conn=self.clipper_conn, 
            model_name=self.clipper_model_name, 
            data=[base64.encodestring(cv2.imencode('.jpg', self.input['img'])[1])],
        )

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
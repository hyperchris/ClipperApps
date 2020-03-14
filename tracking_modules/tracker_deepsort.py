import numpy as np 
import os 
import sys 
from time import time 
from os.path import join 


DS_HOME = join(os.getcwd(), 'tracking_modules/deep_sort')
sys.path.insert(0, DS_HOME)
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching 

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }
'''


class DeepSort:
    input = {}

    def Setup(self):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=4)
        self.log('init')

    def PreProcess(self, input):
        assert input, "DS: empty input!"
        self.input = input

    def Apply(self):
        ''' Extract features and update the tracker 
        ''' 
        detection_list = []
        for obj in self.input['meta']['obj']:
            detection_list.append(Detection(obj['box'], obj['conf'], obj['feature']))

        self.tracker.predict()
        self.tracker.update(detection_list)
        
    def PostProcess(self):
        output = self.input
        output['meta']['obj'] = []
        for tk in self.tracker.tracks:
            if not tk.is_confirmed() or tk.time_since_update > 1:
                continue 
            left, top, width, height = map(int, tk.to_tlwh())
            track_id = tk.track_id
            output['meta']['obj'].append({'box':[left, top, width, height], 
                                            'tid':track_id})
        return output 

    def log(self, s):
        print('[DeepSort] %s' % s)


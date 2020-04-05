import tensorflow as tf
import numpy as np
import logging 

import base64
import cv2 
from clipper_start import register, predict 


GRAPH_PATH = 'checkpoints/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
LABEL_FILE = 'tracking_modules/label_mapping.txt'
PEOPLE_LABEL = 'person'
PEOPLE_THRES = 0.15


class TFDetector(object):
    input = {}

    def Setup(self, graph_path=GRAPH_PATH, label_file=LABEL_FILE):
        self.graph_path = graph_path
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.detection_graph = detection_graph

        with tf.device('/GPU:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(graph=detection_graph, config=config)

        self.label_mapping = {}
        lines = []
        with open(label_file, 'r') as fin:
            lines = fin.readlines()            

        for line in lines:
            line = line.strip()
            if not line or '#' in line:
                continue 
            id, label = line.split(' ')
            self.label_mapping[int(id)] = label

        self.clipper_model_name = 'tracking-detection'
        self.clipper_conn = register(
          model_name=self.clipper_model_name, 
          sess=self.session, 
          func=self.run_session,
        )

        self.log('init')

    def run_session(self, imgs):
        '''
        Runs the object detection on a single image or a batch of images.
        images can be a batch or a single image with batch dimension 1, 
        dims:[None, None, None, 3]

        Args:
        - imgs: a list of encoded images 

        Return:
        - boxes: list of top, left, bottom, right
        - scores: list of confidence 
        - classes: list of labels 
        '''
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        images = np.stack([cv2.imdecode(np.fromstring(base64.b64decode(imgs[0]), dtype=np.uint8), 1)], axis=0)

        (boxes, scores, classes, num_detections) = self.session.run(
                                        [boxes, scores, classes, num_detections],
                                        feed_dict={image_tensor: images[:,:,:,:]})

        return boxes, scores, self.mapping_classes(classes)

    def mapping_classes(self, classes):
        ''' Return the label name of all dets 
        '''
        return [[self.label_mapping[i] if i in self.label_mapping 
                            else str(i) for i in c] for c in classes]

    def PreProcess(self, input):
        assert input, "OBJ_DETECTOR: empty input!"
        self.input = input 

    def Apply(self):
        boxes, scores, classes = predict(
            conn=self.clipper_conn, 
            model_name=self.clipper_model_name, 
            data=[base64.encodestring(cv2.imencode('.jpg', self.input['img'])[1])],
        )
        self.input['meta']['obj'] = []
        for i, b in enumerate(boxes):
            if classes[i] != PEOPLE_LABEL:
                continue 
            if scores[i] < PEOPLE_THRES:
                continue 

            self.input['meta']['obj'].append({
                'box': [int(b[1]), int(b[0]), int(b[3]), int(b[2])],
                'label': classes[i],
                'conf': scores[i],
            })

    def PostProcess(self):
        return self.input

    def log(self, s):
        logging.debug('[TFDetector] %s' % s)


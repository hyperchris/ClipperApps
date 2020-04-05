import numpy as np
from tensorflow.python.framework import tensor_util
import cv2
import tensorflow as tf
import base64
from clipper_start import register, predict 

DEFAULT_MODEL = 'checkpoints/frozen_inference_graph_face.pb'


def load_pb(path_to_pb=DEFAULT_MODEL):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            return detection_graph


def find_face_bounding_box(boxes, scores):
    min_score_thresh = 0.7
    for i in range(0, boxes.shape[0]):
        if scores[i] > min_score_thresh:
            return tuple(boxes[i].tolist())

class FaceDetector:
    def Setup(self):
        # self.graph = load_pb()

        with tf.device('/GPU:0'):
            self.graph = tf.Graph()
            with self.graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(DEFAULT_MODEL, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            self.sess = tf.Session(graph=self.graph, config=config)

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        self.clipper_model_name = 'face-det'
        self.clipper_conn = register(
          model_name=self.clipper_model_name, 
          sess=self.sess, 
          func=self.run_session,
        )
        print('init')

    def PreProcess(self, input):
        self.input = input
        self.image = input['img']
        self.frame_height, self.frame_width= self.image.shape[:2]

    def run_session(self, imgs):
        image_np = cv2.cvtColor(
            cv2.imdecode(np.fromstring(base64.b64decode(imgs[0]), dtype=np.uint8), 1), 
            cv2.COLOR_BGR2RGB
        )
        image_np_expanded = np.expand_dims(image_np, axis=0)
        return self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded}
        )

    def Apply(self):
        (boxes, scores, classes, num_detections) = predict(
            conn=self.clipper_conn, 
            model_name=self.clipper_model_name, 
            data=[base64.encodestring(cv2.imencode('.jpg', self.input['img'])[1])],
        )
        self.bbox = find_face_bounding_box(boxes[0], scores[0])

    def PostProcess(self):
        if self.bbox is None:
            return None

        ymin, xmin, ymax, xmax = self.bbox
        (left, right, top, bottom) = (xmin * self.frame_width, xmax * self.frame_width,
                                      ymin * self.frame_height, ymax * self.frame_height)
        self.input['bounding_box'] = [left, right, top, bottom]        
        return self.input           

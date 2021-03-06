import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2

# sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection')
# from utils import label_map_util
# from utils import visualization_utils as vis_util

# import matplotlib.pyplot as plt
# PATH_TO_LABELS = os.path.join('/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# IMAGE_SIZE = (12, 8)

import base64
from clipper_start import register, predict 


class Inception:
  def Setup(self):

    MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
    # PATH_TO_FROZEN_GRAPH = '/home/yitao/Downloads/tmp/docker-share/module_traffic/models/%s/frozen_inference_graph.pb' % MODEL_NAME
    PATH_TO_FROZEN_GRAPH = 'checkpoints/%s/frozen_inference_graph.pb' % MODEL_NAME   # TODO
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      with tf.device('/GPU:0'):
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
        if 'detection_masks' in self.tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image.shape[1], image.shape[2])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          self.tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
        self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    self.clipper_model_name = 'traffic-inception'
    self.clipper_conn = register(
      model_name=self.clipper_model_name, 
      sess=self.sess, 
      func=self.run_session,
    )
    print(self.clipper_model_name, 'READY')

  def run_session(self, imgs):
    input = np.expand_dims(cv2.imdecode(np.fromstring(base64.b64decode(imgs[0]), dtype=np.uint8), 1), axis=0)
    return self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: input})

  def PreProcess(self, input):
    self.output = input
    self.input = [base64.encodestring(cv2.imencode('.jpg', input['img'])[1])]


  def Apply(self):
    output_dict = predict(
      conn=self.clipper_conn, 
      model_name=self.clipper_model_name, 
      data=self.input,
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    self.output['num_detections'] = int(output_dict['num_detections'][0])
    self.output['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    self.output['detection_boxes'] = output_dict['detection_boxes'][0]
    self.output['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      self.output['detection_masks'] = output_dict['detection_masks'][0]

  def PostProcess(self):
    return self.output


unit_test = False
if (unit_test):
  myInception = Inception()
  myInception.Setup()

  image_np = cv2.imread("/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection/test_images/image1.jpg")
  # image_np_expanded = np.expand_dims(image_np, axis=0)
  myInception.PreProcess(image_np)
  myInception.Apply()
  output_dict = myInception.PostProcess()

  print(output_dict)

# vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks'),
#       use_normalized_coordinates=True,
#       line_thickness=8)
# plt.figure(figsize=IMAGE_SIZE)
# plt.imshow(image_np)
# plt.show()
# plt.savefig('tmp.png')

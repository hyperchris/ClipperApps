import numpy as np

from skimage.transform import estimate_transform
import cv2
from tensorflow.python.framework import tensor_util
import tensorflow as tf


class PRNetImageCropper:
    def Setup(self):
        self.resolution_inp = 256
        self.DST_PTS = np.array(
            [[0, 0], 
            [0, self.resolution_inp - 1], 
            [self.resolution_inp - 1, 0]]
        )
         
    def PreProcess(self, face_data):
        self.output = face_data
        bounding_box = face_data["bounding_box"]
        self.image = face_data["img"]
        
        left = bounding_box[0]
        right = bounding_box[1]
        top = bounding_box[2]
        bottom = bounding_box[3]
        
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0,
                           bottom - (bottom - top) / 2.0 + old_size * 0.14])
        size = int(old_size * 1.58)
        self.src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] -
                                                                           size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
    
    def Apply(self):
        self.tform = estimate_transform('similarity', self.src_pts, self.DST_PTS)
        image = self.image / 255.
        self.cropped_image = cv2.warpAffine(
            image, 
            self.tform.params[:2], 
            dsize=(self.resolution_inp, self.resolution_inp)
        )

    def PostProcess(self):
        res = self.output
        res['cropped_image'] = self.cropped_image
        res['tform_params'] = self.tform.params
        res['img'] = self.image

        return res 

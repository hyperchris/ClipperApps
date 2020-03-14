import numpy as np
import tensorflow as tf
import os 
import time
from tensorflow.python.framework import tensor_util
from face_modules.predictor import PosPrediction

import base64
import cv2
import json
import requests

def clipper_query(addr, img):
    url = "http://%s/face2-endpoint/predict" % addr
    retval, buf = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buf)
    req_json = json.dumps({
        "input":
        jpg_as_text.decode() # bytes to unicode
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)


DEFAULT_MODEL = 'checkpoints/256_256_resfcn256_weight'

class PRNet:
    def Setup(self):
        self.resolution_inp = 256
        self.resolution_op = 256
        self.MaxPos = self.resolution_inp * 1.1
        prefix='.'

        self.uv_kpt_ind = np.loadtxt(
            prefix + '/face_modules/PRNet/Data/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(
            prefix + '/face_modules/PRNet/Data/uv-data/face_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt(
            prefix + '/face_modules/PRNet/Data/uv-data/triangles.txt').astype(np.int32)
        
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        # assert os.path.exists(DEFAULT_MODEL), "model not exists!"
        self.pos_predictor.restore(DEFAULT_MODEL)

    def PreProcess(self, prnet_data):
        self.input = prnet_data
        self.image = prnet_data["img"]
        self.tform_params = prnet_data["tform_params"]
        self.cropped_image = prnet_data["cropped_image"]

    def Apply(self):
        self.cropped_pos = self.pos_predictor.predict(self.cropped_image)

    def PostProcess(self):
        cropped_vertices = np.reshape(self.cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / self.tform_params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(self.tform_params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(
            vertices.T, [self.resolution_op, self.resolution_op, 3])

        key_points = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        self.input["vertices"] = vertices
        return self.input

import sys 
from threading import Thread

from face_modules.face_detector import FaceDetector
from face_modules.prnet_image_cropper import PRNetImageCropper
from face_modules.prnet import PRNet
from support import thread_func, Q, FPSCounter, DataReader


# ============ Video Input Modules ============
reader = DataReader()
fps = 10 if len(sys.argv) < 3 else int(sys.argv[2]) 
reader.Setup(video_path=sys.argv[1], fps=fps)

# ============ Face Detection Modules ============
face_detector = FaceDetector()
face_detector.Setup()

# ============ PRNET Modules ============
prnet_image_cropper = PRNetImageCropper()
prnet_image_cropper.Setup()

prn = PRNet()
prn.Setup()

# ============ Queues ============
buf_size = 64
img_queue = Q(buf_size)
det_queue = Q(buf_size)
crop_queue = Q(buf_size)
res_queue = Q(buf_size)

# ============ Start main threads ============
fps_counter = FPSCounter()

img_th = Thread(target=thread_func, args=("reader", reader, None, img_queue))
det_th = Thread(target=thread_func, args=("face_detector", face_detector, img_queue, det_queue))
crop_th = Thread(target=thread_func, args=("face_cropper", prnet_image_cropper, det_queue, crop_queue))
res_th = Thread(target=thread_func, args=("prnet", prn, crop_queue, res_queue))
fps_th = Thread(target=thread_func, args=("fps", fps_counter, res_queue, None))

th_pool = [img_th, det_th, crop_th, res_th, fps_th][::-1]

for th in th_pool:
    th.daemon = True 
    th.start() 

_ = raw_input("Enter anykey to quit: ")
print("done")
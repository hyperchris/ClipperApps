import sys 
from threading import Thread

from tracking_modules.object_detector_tf import TFDetector
from tracking_modules.reid_extractor import FeatureExtractor
from tracking_modules.tracker_deepsort import DeepSort 
from support import thread_func, Q, FPSCounter, DataReader


# ============ Video Input Modules ============
reader = DataReader()
fps = 10 if len(sys.argv) < 3 else int(sys.argv[2]) 
reader.Setup(video_path=sys.argv[1], fps=fps)

# ============ Object Detection Modules ============
object_detector = TFDetector()
object_detector.Setup()

# ============ Tracking Modules ============
feature_extractor = FeatureExtractor()
feature_extractor.Setup()

deepsort = DeepSort()
deepsort.Setup()

tracker = deepsort

# ============ Queues ============
buf_size = 64
img_queue = Q(buf_size)
det_queue = Q(buf_size)
feature_queue = Q(buf_size)
res_queue = Q(buf_size)

# ============ Start main threads ============
fps_counter = FPSCounter()

img_th = Thread(target=thread_func, args=("reader", reader, None, img_queue))
det_th = Thread(target=thread_func, args=("object_detector", object_detector, img_queue, det_queue))
feature_th = Thread(target=thread_func, args=("feature_extractor", feature_extractor, det_queue, feature_queue))
res_th = Thread(target=thread_func, args=("tracker", tracker, feature_queue, res_queue))
fps_th = Thread(target=thread_func, args=("fps", fps_counter, res_queue, None))

th_pool = [img_th, det_th, feature_th, res_th, fps_th][::-1]

for th in th_pool:
    th.daemon = True 
    th.start() 

_ = raw_input("Enter anykey to quit: ")
print("done")
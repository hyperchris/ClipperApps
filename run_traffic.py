import sys 
from time import sleep 
from threading import Thread

from traffic_modules.traffic_yolo_tf import YOLO
from traffic_modules.inception_tf import Inception
from traffic_modules.resnet_tf import Resnet
from support import thread_func, thread_func2, Q, FPSCounter, DataReader


# ============ Video Input Modules ============
reader = DataReader()
fps = 10 if len(sys.argv) < 3 else int(sys.argv[2]) 
reader.Setup(video_path=sys.argv[1], fps=fps)

# ============ Modules ============
yolo = YOLO()
yolo.Setup()

inception = Inception()
inception.Setup()

resnet = Resnet()
resnet.Setup()

# ============ Queues ============
buf_size = 2
img_queue = Q(buf_size)
yolo_queue1 = Q(buf_size)
yolo_queue2 = Q(buf_size) 
inc_queue = Q(buf_size)
res_queue = Q(buf_size)

# ============ Start main threads ============
fps_counter1, fps_counter2 = FPSCounter("Inception:"), FPSCounter("Resnet:") 

img_th = Thread(target=thread_func, args=("reader", reader, None, img_queue))
yolo_th = Thread(target=thread_func2, args=("yolo", yolo, img_queue, yolo_queue1, yolo_queue2))
inc_th = Thread(target=thread_func, args=("inception", inception, yolo_queue1, inc_queue))
res_th = Thread(target=thread_func, args=("resnet", resnet, yolo_queue2, res_queue))
fps_th1 = Thread(target=thread_func, args=("fps-1", fps_counter1, inc_queue, None))
fps_th2 = Thread(target=thread_func, args=("fps-2", fps_counter2, res_queue, None))

th_pool = [img_th, yolo_th, inc_th, res_th, fps_th1, fps_th2][::-1]

for th in th_pool:
    th.daemon = True 
    th.start() 

_ = input("Enter anykey to quit: ")
print("done")

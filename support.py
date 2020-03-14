import os 
import cv2 
import numpy as np
import sys 
import Queue
from time import sleep, time


''' Read video and metadata file (opt) and output the data sequence

Input: 
    - Video file path
    - Optional: metadata file path (an npy file)
            format: [{'frame_id': frame_id, xxx}]
                xxx means you can put whatever k-v entry
Output: 
    {'img':img_np_array, 'time': time_of_input, 'meta':{'frame_id':frame_id}}

'''
class DataReader:
    src = ''
    cap = None
    data = [] 
    frame_id = 0
    read_time_gap = 0.1
    last_read_time = time() 
    end_of_video = False 


    def Setup(self, video_path='', fps=10, file_path=''):
        if not video_path.isdigit() and not os.path.exists(video_path):
            self.log('Cannot load video!')
            self.end_of_video = True
            return 
        
        self.src = int(video_path) if self.src.isdigit() else video_path
        self.cap = cv2.VideoCapture(video_path)

        self.data_ptr = 0 
        if file_path and os.path.exists(file_path):
            self.data = np.load(open(file_path, 'r'))
        
        self.read_time_gap = 1. / fps
        self.log('init')

    def PreProcess(self, data): 
        pass 

    def Apply(self):
        pass

    def PostProcess(self):
        if self.end_of_video:
            return None

        sleep(max(0, self.read_time_gap - (time() - self.last_read_time)))
        self.last_read_time = time() 
        ret, frame = self.cap.read()

        if not ret:
            self.log('End of video')
            self.end_of_video = True 
            return None

        output = {'img': frame, 'time': time(), 'meta': {'frame_id':self.frame_id}}
        while len(self.data) and self.data_ptr < len(self.data) and \
                self.data[self.data_ptr]['frame_id'] < self.frame_id:
            self.data_ptr += 1

        if len(self.data) and self.data_ptr < len(self.data) and \
                self.data[self.data_ptr]['frame_id'] == self.frame_id:
            output['meta'] = self.data[self.data_ptr]

        self.frame_id += 1
        return output 


    def log(self, s):
        print('[DataReader] %s' % s)


class Q:
    """
    A class for queue 
    """
    def __init__(self, qsize):
        self.q = Queue.Queue(qsize)

    def write(self, v):
        try:
            self.q.put_nowait(v)
        except Queue.Full:
            pass

    def read(self):
        try:
            d = self.q.get_nowait()
            return d 
        except Queue.Empty:
            return None

    def empty(self):
        return self.q.empty()

    def full(self):
        return self.q.full()

    def qsize(self):
        return self.q.qsize()


class FPSCounter:

    def __init__(self, name=""):
        self.timer = time()
        self.name = name 

    def PreProcess(self, data):
        if 'time' in data:
            print(self.name + " Latency: %0.1fs" % (time() - data['time']))

    def Apply(self):
        print(self.name + "FPS: %0.1f" % (1. / (time() - self.timer)))
        self.timer = time()  

    def PostProcess(self):
        return None


def thread_func(name, target, in_queue, out_queue):
    """
    Starts a thread that keeps running a function for processing data 
    from in_queue and writing the results into out_queue
    """
    print("Starting thread %s..." % name)
    while True: 
        if in_queue is not None:
            data = in_queue.read()
            if data is None:
                sleep(0.01)
                continue 
            target.PreProcess(data)
            target.Apply()    

        res = target.PostProcess()
        if res is not None and out_queue is not None:
            out_queue.write(res)


def thread_func2(name, target, in_queue, out_queue1, out_queue2):
    """
    Starts a thread that keeps running a function for processing data 
    from in_queue and writing the results into two out_queues
    """
    print("Starting thread %s..." % name)
    while True: 
        if in_queue is not None:
            data = in_queue.read()
            if data is None:
                sleep(0.01)
                continue 
            target.PreProcess(data)
            target.Apply()    

        res = target.PostProcess()
        if res is not None:
            out_queue1.write(res)
            out_queue2.write(res)            

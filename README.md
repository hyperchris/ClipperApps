# ClipperApps
TF applications using Clipper 

## Recommended Environment
- Python 2
- Ubuntu 16.04
- TensorFlow 1.12.0
- CUDA 9.0 and compatible CuDNN

## How to Run
- Clone this repo 
```
git clone https://github.com/hyperchris/ClipperApps.git
```
- Download these two folders, [data](https://drive.google.com/drive/folders/1M8Ct0H1IdYKnmA-PFU87st_Un0e9rWws?usp=sharing) and [checkpoints](https://drive.google.com/drive/folders/1KInJxEvzH6eppyBDEWdhHp3GFVpN26DT?usp=sharing) under ```ClipperApps/```
- Install packages
```
pip install -r requirements.txt
```
- Run the setup script then start Clipper. Keep Clipper in "started" status.
```
sudo bash setup.sh; python clipper_start.py
```
- Open another terminal to run three chains: face, tracking, and traffic. You can run a chain with this command
```
python run_tracking.py data/tracking.mp4
```
Once a chain starts, it will output its FPS and latency in terminal. Press ```Ctrl-C``` or ```Anykey + Enter``` to quit. 

## Evaluation
There are three options to measure GPU util:
1. ```python gpu_monitor.py```
2. ```python gpu_monitor2.py```
3. ```nvidia-smi -l 1```

**Specifiy GPU to Use**
Change the following lines
- Traffic chain: [YOLO](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/traffic_modules/traffic_yolo_tf.py#L21), [Inception](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/traffic_modules/inception_tf.py#L37), [ResNet](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/traffic_modules/resnet_tf.py#L37)
- Tracking chain: [ObjectDetection](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/tracking_modules/object_detector_tf.py#L31), [TrackingReID](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/tracking_modules/deep_sort/tools/generate_detections.py#L75)
Face chain: [FaceDetection](https://github.com/hyperchris/ClipperApps/blob/d54981ebe0abd80047ed375d29a88421d8fded1f/face_modules/face_detector.py#L32)

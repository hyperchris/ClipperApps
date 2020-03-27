from __future__ import print_function
import json
import requests
import time
import signal
import os
import sys


sys.path.insert(1, os.getcwd() + '/clipper/clipper_admin')
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model
from clipper_admin.deployers.tensorflow import tensorflow_predict


def predict(conn, model_name, data):
    """
    run the prediction function 

    Input: 
    - conn: clipper connection 
    - model_name: string 
    - data: a list of data as strings  

    Return: 
    - custom func output
    """
    if not isinstance(data, list):
        print("The input data should be a list")
        return None 
    req_json = json.dumps({'model': model_name, 'input': data})
    headers = {'Content-type': 'application/json'}
    return tensorflow_predict(
        clipper_conn=conn,
        headers=headers, 
        req=req_json,
    )


def register(model_name, sess, func):
    """
    Register a tf session with its function 

    Input: 
    - model_name: name of the model, string
    - sess: TF session
    - func: the function that runs the TF session 

    Return:
    - clipper connection 
    """
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.connect()
    deploy_tensorflow_model(
        clipper_conn=clipper_conn,
        name=model_name,
        version='1.0',
        input_type='strings',
        func=func,
        tf_sess_or_saved_model_path=sess,
    )
    print(model_name, "registered")
    return clipper_conn


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
    print("Stopping Clipper...")
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    clipper_conn = ClipperConnection(DockerContainerManager(use_centralized_log=False))
    clipper_conn.stop_all()
    clipper_conn.start_clipper()
    print('Clipper Started')
    try:
        while True:
            time.sleep(2)
    except Exception as e:
        clipper_conn.stop_all()

    print("done")
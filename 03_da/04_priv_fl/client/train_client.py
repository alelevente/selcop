from flask import Flask, request
from flask.logging import default_handler
import logging

import logging
import gc

import json
import pickle
import os, signal

import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")
import neural_network


with open("/data/edge_maps.json") as f:
    edge_map_dict = json.load(f)
    edge_map = edge_map_dict["edge_to_idx"]
    idx_to_edge = edge_map_dict["idx_to_edge"]
    
    
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


app = Flask(__name__)
log_level = logging.CRITICAL
app.logger.setLevel(log_level)
app.logger.disabled = True #no logging is needed
logging.getLogger('werkzeug').disabled = True #logging disabled


#setting up the GPU and the NN:
logical_gpu = [tf.config.LogicalDeviceConfiguration(memory_limit=1280)]
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        logical_gpu
    )
except:
    pass

devices = tf.config.list_logical_devices("GPU")
if len(devices) == 0:
    gpu = tf.config.list_logical_devices()[0]
else:
    gpu = devices[0]
    
with tf.device(gpu):        
    nn = neural_network.NeuralNetwork()


def do_training(payload):
    #parsing payload:
    train_features, train_labels = np.array(payload["train_features"]), np.array(payload["train_labels"])
    weights_decoded = neural_network.decode_weights(payload["model_weights"])
    epochs = payload["epochs"]

    test_results = {}    

    #training:
    with tf.device(gpu):
        nn.model.set_weights(weights_decoded)
    
        #print("training", flush=True)
        history = nn.train(train_features, train_labels, epochs = epochs)
        weight_updates = neural_network.encode_weights(nn.model.get_weights())
        
        #print("testing", flush=True)
        timing = []
        for i in range(100):
            for t in range(120):
                timing.append(t/(24*60))
        p = 0
        while p<len(edge_map):
            test_data_p = np.zeros((100*120, len(edge_map)+1)).astype(np.float32)
            i = 0
            while i<100 and p+i < len(edge_map):
                test_data_p[i*120: (i+1)*120, p+i] = 1.0
                i += 1
            p += i
            
            test_data_p[:,-1] = timing
            infered = nn.model.predict(test_data_p, batch_size=10000, verbose=0).tolist()
            #print(len(infered), flush=True)
            for i in range(100):
                if p+i<len(idx_to_edge):
                    idx = idx_to_edge[f"{p+i}"]
                    test_results[idx] = infered[i*120: (i+1)*120]

    answer = {"model_update": weight_updates,
            "n_samples": len(train_features),
            "test_results": test_results} 
    #print(answer, flush=True)
    return answer

@app.route("/compute", methods=["POST"])
def compute():
    payload = json.loads(request.get_data(), strict=False)
    answer = do_training(payload)
    gc.collect()
    return json.dumps(answer)

@app.route('/shutdown', methods=['GET'])
def stopServer():
    print("terminating")
    os.kill(os.getpid(), signal.SIGTERM)
    return json.dumps({})


if __name__ == "__main__":
    port = os.environ["TRAIN_PORT"]
    app.run(host="0.0.0.0", port = port)
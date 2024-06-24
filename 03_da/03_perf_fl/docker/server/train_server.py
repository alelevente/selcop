import numpy as np
import pandas as pd

import json
import pickle
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import neural_network

import requests
from multiprocessing.pool import Pool
import copy

from tqdm import trange


SEEDS = [42, 1234, 1867, 613, 1001]
SHARING_METHODS = ["ref", "all_data", "alters", "zopt"]

SHARED_DATA_ROOT = "/meeting_data"
RESULT_PATH = "/results"
SIMULATION_PATH = "/simulation_results"

BASE_ADDRESS = "http://train"
CONFIGURATION = "training_configuration.json"

NUM_CLIENTS = 6
VEHICLES_PER_COMM_ROUND = 6

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open("/data/one_hot_encoding_dict.json") as f:
    oh_encoding_dict = json.load(f)
    

def train_client(args):
    day = args["day"] + 4
    vehicle = args["vehicle"]
    method = args["sharing_method"]

    vehicle_data = pd.DataFrame()
    for s in SEEDS:
        datasource = f"{SHARED_DATA_ROOT}/{method}/{s}/{vehicle}.csv"
        pf = pd.read_csv(datasource)
        pf["seed"] = [s]*len(pf)
        vehicle_data = pd.concat([vehicle_data, pf])

    true_parkings = vehicle_data["parking_id"].unique()
    train_features, train_labels = neural_network.prepare_train_data(vehicle_data,
                                                                     day*24*60*60,
                                                                     (day+1)*24*60*60,
                                                                     parking_map)
    
    if len(train_features) == 0:
        return {"model_update": [],
                "n_samples":0}

    payload = {
        "train_features": neural_network.encode_weights(train_features),
        "train_labels": train_labels.tolist(),
        "model_weights": args["model_weights"],
        "epochs": 1
    }
    try:
        r = requests.post(args["address"], json=payload)

        response = json.loads(r.text)

        return {
            "federated_results": json.loads(r.text),
            "vehicle": vehicle
        }
    except:
        return {"model_update": [],
                "n_samples":0}


def fed_avg(model_weights, samples):
    layers = []
    for l in range(len(model_weights[0])):
        layer_weights = []
        for update in model_weights:
            layer_weights.append(update[l])
        layer_weights = np.array(layer_weights)
        layers.append(np.average(layer_weights, axis=0, weights=samples))

    return layers


def combine_commuters(veh_id):
    if veh_id.startswith("carIn"):
        return veh_id.split(":")[0]
    return veh_id

def _prepare_eval_data(dataset, min_time, max_time, parking_map):
    dataset = dataset[dataset["time"] >= min_time]
    dataset = dataset[dataset["time"] < max_time].copy()
    dataset["time_of_day"] = dataset["time"] % (24*60*60)
    dataset["time_of_day"] = dataset["time_of_day"] / (24*60*60)
    labels = dataset["occupancy"].copy()

    features = np.zeros((len(dataset), len(parking_map)+1)) #one-hot-encoded parking_id | time_of_day
    i = 0
    for _,r in dataset.iterrows():
        features[i,parking_map[r["parking_id"]]] = 1.0
        i += 1
    features[:,-1] = dataset["time_of_day"]
    return features, labels

    
if __name__ == "__main__":
    addresses = []
    #time.sleep(5) #waiting for other containers
    with open(CONFIGURATION) as f:
        train_config = json.load(f)
    addresses = train_config["addresses"]
    

    with open("/data/veh_list.json") as f:
        veh_map = json.load(f)
    test_vehicles = veh_map["test_vehs"]

    with open("/data/parking_map.json") as f:
        parking_map = json.load(f)

    #FL evaluation data:
    p_data = pd.DataFrame()
    #READING DATA:
    for s in SEEDS:
        filename =f"{SIMULATION_PATH}/poccup_by_vehs_{s}.csv"
        pf = pd.read_csv(filename)
        pf["seed"] = [s]*len(pf)
        p_data = pd.concat([p_data, pf])
        
    

    #last day is the evaluation day:
    p_data = p_data[p_data["time"] > min(p_data["time"])+4*24*60*60]
    p_data["veh_id"] = p_data["veh_id"].apply(combine_commuters)
    X_eval, y_eval = _prepare_eval_data(p_data,
                                        min_time=min(p_data["time"]),
                                        max_time=max(p_data["time"]),
                                        parking_map=parking_map)

    
    logical_gpu = [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]
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

    
    for sm in SHARING_METHODS:
        #creating new NN-model for each sharing method:
        with tf.device(gpu):
            nn = neural_network.NeuralNetwork()

            for day in range(0,4):
                print(f"===================== DAY {day} ======================")
                #selecting vehicles to train:
                inference_results = {}
                sel_vehicles = np.random.choice(test_vehicles, 250, replace=False)
                #for computational reasons, only a part of the test vehicles will be selected in a comm round:
                for veh_range in trange(0, len(sel_vehicles), VEHICLES_PER_COMM_ROUND):
                    print(f"Vehicles: [{veh_range}: {veh_range+VEHICLES_PER_COMM_ROUND})")
                    model_weights, num_samples = [],[]
                    
                    weights = neural_network.encode_weights(nn.model.get_weights())
                    arguments = []
                    i = 0
                    for vehicle in sel_vehicles[veh_range:(veh_range+VEHICLES_PER_COMM_ROUND)-1]:
                        address = i%len(addresses)
                        arguments.append({
                            "day": day,
                            "vehicle": vehicle,
                            "address": addresses[address],
                            "model_weights": weights,
                            "sharing_method": sm
                        })
                        i += 1

                    with Pool(NUM_CLIENTS) as pool:
                        results = pool.map(train_client, arguments)
                        for r in results:
                            if ("federated_results" in r) and (r["federated_results"]["n_samples"] != 0):
                                #inputs for the federated learning:
                                mod_weights = neural_network.decode_weights(r["federated_results"]["model_update"])
                                if len(mod_weights)>0:
                                    model_weights.append(mod_weights)
                                    num_samples.append(r["federated_results"]["n_samples"])

                    if len(model_weights)>0: #it happens when a handful of randUni vehicles are selected
                        updated_weights = fed_avg(model_weights, num_samples)
                        nn.model.set_weights(updated_weights)
                    else:
                        pass

                #evaluating federated model
                test_val = nn.model.evaluate(x=X_eval, y=y_eval, batch_size=10000, verbose=False)
                if day != 0:
                    #reading the already existing file:
                    with open(f"/{RESULT_PATH}/{sm}/fl_eval_performance.json") as f:
                        fl_perf = json.load(f)
                else:
                    #creating the file:
                    fl_perf = {"fl_performance": []}
                fl_perf["fl_performance"].append(
                    {"evaluation_result": test_val,
                     "day": day})
                with open(f"/{RESULT_PATH}/{sm}/fl_eval_performance.json", "w") as f:
                    json.dump(fl_perf, f)

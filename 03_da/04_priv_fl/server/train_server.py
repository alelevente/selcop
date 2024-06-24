import numpy as np
import pandas as pd

import json
import pickle
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import neural_network
import inference_methods

import requests
from multiprocessing.pool import Pool
import copy

from tqdm import trange


SEEDS = [42, 1234, 1867, 613, 1001]
SHARED_DATA_ROOT = "/meeting_data"
RESULT_PATH = "/results"
SIMULATION_PATH = "/simulation_results"

SHARING_METHODS = ["ref", "all_data", "alters", "zopt"]
BASE_ADDRESS = "http://train"
CONFIGURATION = "training_configuration.json"

NUM_CLIENTS = 7
VEHICLES_PER_COMM_ROUND = 7

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def combine_commuters(veh_id):
    if veh_id.startswith("carIn"):
        return veh_id.split(":")[0]
    return veh_id

with open("/data/one_hot_encoding_dict.json") as f:
    oh_encoding_dict = json.load(f)

#################################################################
############## PRETRAINING A GLOBAL MODEL #######################
def pretrain_model(parking_map, vehicles):
    
    def _sample_a_batch(X, y, portion=1.0, parking_map=None):
        train_indices = np.random.randint(0, len(X), int(len(X)*portion))
        x_batch = X.iloc[train_indices]
        y_batch = y.iloc[train_indices]
        
        features = np.zeros((len(x_batch), len(parking_map)+1))
        j = 0
        for i,r in x_batch.iterrows():
            features[j,oh_encoding_dict[r["parking_id"]]]=1.0
            features[j,-1] = r["time_of_day"]
            j += 1
        
        return features, np.array(y_batch)
    
    p_data = pd.DataFrame()
    #READING DATA:
    for s in SEEDS:
        filename = f"{SIMULATION_PATH}/poccup_by_vehs_{s}.csv"
        pf = pd.read_csv(filename)
        pf["seed"] = [s]*len(pf)
        p_data = pd.concat([p_data, pf])
        
    p_data["veh_id"] = p_data["veh_id"].apply(combine_commuters)
    parkings = p_data["parking_id"].unique()
    
    p_data["time"] = p_data["time"] - 4*24*60*60
    p_data["time"] = p_data["time"].astype(int)
    p_data["time_of_day"] = (p_data["time"] - (p_data["time"] // (24*60*60))*24*60*60) / (24*60*60) #converting to 0.0-1.0 and removing periodicity
           
    #creating datasets:
    p_train = p_data[p_data["veh_id"].isin(vehicles)]

    X_train = p_train.drop(columns=["veh_id", "time", "occupancy", "seed"])
    y_train = p_train["occupancy"]
    
    train_indices = np.random.randint(0, len(X_train), 1000000)
    x_train_batch = X_train.iloc[train_indices]
    y_train_batch = y_train.iloc[train_indices]
    x_train_batch, y_train_batch = _sample_a_batch(x_train_batch, y_train_batch, parking_map=parking_map)
    
    with tf.device('/CPU:0'):
        nn = neural_network.NeuralNetwork()
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)]

        history = nn.model.fit(x=x_train_batch, y=y_train_batch, epochs=5, batch_size=10000, callbacks=callbacks)
        while len(history.history["loss"])%5 == 0:
            history = nn.model.fit(x=x_train_batch, y=y_train_batch, epochs=5, batch_size=10000, callbacks=callbacks)
            
    return nn.model
#################################################################
        
def _generate_test_data_for_p(parking):
    test_data_p = np.zeros((24*60*60, len(oh_encoding_dict)+1)).astype(np.float32)
    i = oh_encoding_dict[parking]
    test_data_p[:,i] = 1.0
    test_data_p[:,-1] = np.linspace(0.0, 1.0, len(test_data_p))
    return test_data_p

def train_client(args):
    vehicle = args["vehicle"]
    method = args["sharing_method"]
    global_predictions = args["global_predictions"]

    vehicle_data = pd.DataFrame()
    for s in SEEDS:
        datasource = f"{SHARED_DATA_ROOT}/{method}/{s}/{vehicle}.csv"
        pf = pd.read_csv(datasource)
        pf["seed"] = [s]*len(pf)
        vehicle_data = pd.concat([vehicle_data, pf])

    own_data = vehicle_data[vehicle_data["receive_time"] == -1]
    true_parkings = own_data["parking_id"].unique()
    train_features, train_labels = neural_network.prepare_train_data(vehicle_data,
                                                                     4*24*60*60,
                                                                     10*24*60*60,
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

        #location inference:
        vehicle_predictions = response["test_results"]

        p_diffs = inference_methods.create_difference_dataset(global_predictions, vehicle_predictions)
        pred_lots = inference_methods.predict_eval_positions(p_diffs, true_parkings)
        offset = inference_methods.predict_eval_time(p_diffs, train_features[:,-1])

        inferenced = {
            "positions": list(pred_lots),
            "time_offset": offset
        }

        return {
            "inference_results": inferenced,
            "vehicle": vehicle
        }
    except:
        return {"inference_results": [],
                "vehicle": vehicle}


def fed_avg(model_weights, samples):
    layers = []
    for l in range(len(model_weights[0])):
        layer_weights = []
        for update in model_weights:
            layer_weights.append(update[l])
        layer_weights = np.array(layer_weights)
        layers.append(np.average(layer_weights, axis=0, weights=samples))

    return layers

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
    with open(CONFIGURATION) as f:
        train_config = json.load(f)
    addresses = train_config["addresses"]
    

    with open("/data/veh_list.json") as f:
        vehs = json.load(f)
        
    adversarial_targets = np.random.choice(vehs["test_vehs"], 300, replace=False)
    adversarial_global_trainers = np.random.choice(list(set(vehs["train_vehs"]+vehs["test_vehs"]).difference(adversarial_targets)), 3700, replace=False)
    test_vehicles = adversarial_targets

    with open("/data/parking_map.json") as f:
        parking_map = json.load(f)

    #pretraining a global data:
    global_pretrained = pretrain_model(parking_map, adversarial_global_trainers)
    
    #prepare baseline for inference methods:
    global_predictions = {}
    #print("Computing reference predictions.")
    with tf.device('/CPU:0'):
        for p in parking_map:
            test_data_p = _generate_test_data_for_p(p)
            global_predictions[p] = global_pretrained.predict(test_data_p, batch_size=10000, verbose=0)

    weights = neural_network.encode_weights(global_pretrained.get_weights())

    for sm in SHARING_METHODS:
        #creating new NN-model for each sharing method:
        with tf.device('/CPU:0'):         
            inference_results = {}
            for veh_range in trange(0, len(test_vehicles), VEHICLES_PER_COMM_ROUND):

                arguments = []
                i = 0
                for vehicle in test_vehicles[veh_range:(veh_range+VEHICLES_PER_COMM_ROUND)-1]:
                    address = i%len(addresses)
                    arguments.append({
                        "vehicle": vehicle,
                        "address": addresses[address],
                        "model_weights": weights,
                        "sharing_method": sm,
                        "global_predictions": global_predictions
                    })
                    i += 1

                #print("Dataset prepared. Start training.")
                with Pool(NUM_CLIENTS) as pool:
                    results = pool.map(train_client, arguments)
                    for r in results:
                        if ("inference_results" in r):
                            inference_results[r["vehicle"]] = r["inference_results"]


            with open(f"/{RESULT_PATH}/{sm}/inference_results.json", "w") as f:
                json.dump(inference_results, f)
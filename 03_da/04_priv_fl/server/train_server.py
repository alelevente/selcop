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


TRAIN_SEEDS = [42, 1234, 1867, 613, 1001, 704, 882, 405]
TEST_SEEDS = [269, 120]
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

with open("/data/edge_maps.json") as f:
    edge_map = json.load(f)["edge_to_idx"]

#################################################################
############## PRETRAINING A GLOBAL MODEL #######################
def pretrain_model(edge_map, vehicles):
    
    def _sample_a_batch(X, y, portion=1.0, edge_map=None):
        train_indices = np.random.randint(0, len(X), int(len(X)*portion))
        x_batch = X.iloc[train_indices]
        y_batch = y.iloc[train_indices]
        
        features = np.zeros((len(x_batch), len(edge_map)+1))
        j = 0
        for i,r in x_batch.iterrows():
            if r["edge"] in edge_map:
                features[j,edge_map[r["edge"]]]=1.0
                features[j,-1] = r["time_of_day"]
            j += 1
        
        return features, np.array(y_batch)
    
    combined_df = pd.read_csv(f"{SHARED_DATA_ROOT}/combined_dataset.csv")
    combined_df = combined_df[combined_df["seed"].isin(TRAIN_SEEDS)]
    combined_df["time_of_day"] = combined_df["time"] / (24*60*60)
           
    #creating datasets:
    combined_df = combined_df[combined_df["veh_id"].isin(vehicles)]

    X_train = combined_df.drop(columns=["veh_id", "time", "rel_speed", "seed", "hash"])
    y_train = combined_df["rel_speed"]
    
    train_indices = np.random.randint(0, len(X_train), 1000000)
    x_train_batch = X_train.iloc[train_indices]
    y_train_batch = y_train.iloc[train_indices]
    x_train_batch, y_train_batch = _sample_a_batch(x_train_batch, y_train_batch, edge_map=edge_map)
    
    with tf.device('/CPU:0'):
        nn = neural_network.NeuralNetwork()
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)]

        history = nn.model.fit(x=x_train_batch, y=y_train_batch, epochs=5, batch_size=10000, callbacks=callbacks)
        epoch = 0
        while (len(history.history["loss"])%5 == 0) and (epoch<50):
            history = nn.model.fit(x=x_train_batch, y=y_train_batch, epochs=5, batch_size=10000, callbacks=callbacks)
            epoch += 5
            
    return nn.model
#################################################################
        
def _generate_test_data_for_p(edge):
    test_data_p = np.zeros((2*60, len(edge_map)+1)).astype(np.float32)
    i = edge_map[edge]
    test_data_p[:,i] = 1.0
    test_data_p[:,-1] = np.linspace(0.0, (24*60*60)/7200.0+1, len(test_data_p))
    return test_data_p

def train_client(args):
    vehicle = args["vehicle"]
    method = args["sharing_method"]
    global_predictions = args["global_predictions"]

    #print("client loads data", flush=True)
    vehicle_data = pd.DataFrame()
    for s in TRAIN_SEEDS:
        datasource = f"{SHARED_DATA_ROOT}/{method}/{s}/{vehicle}.csv"
        pf = pd.read_csv(datasource)
        pf["seed"] = [s]*len(pf)
        vehicle_data = pd.concat([vehicle_data, pf])

    own_data = vehicle_data[vehicle_data["receive_time"] == -1]
    true_edges = own_data["edge"].unique()
    #print("client prepares dataset", flush=True)
    train_features, train_labels = neural_network.prepare_train_data(vehicle_data, edge_map)
    
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
        #print("sending training request", flush=True)
        r = requests.post(args["address"], json=payload)

        #print("response received", flush=True)
        response = json.loads(r.text)

        #location inference:
        #print(f"response: {response}", flush=True)
        vehicle_predictions = response["test_results"]
        #print(f"response resutls: {list(vehicle_predictions.keys())[:10]}", flush=True)

        e_diffs = inference_methods.create_difference_dataset(global_predictions, vehicle_predictions)
        #print("difference calculations complete")
        pred_lots = inference_methods.predict_eval_positions(e_diffs, true_edges)
        offset = inference_methods.predict_eval_time(e_diffs, train_features[:,-1])

        inferenced = {
            "positions": list(pred_lots),
            "time_offset": offset
        }
        #print(inferenced, flush=True)

        return {
            "inference_results": inferenced,
            "vehicle": vehicle
        }
    except Exception as e:
    #    print(f"error occured: {e}", flush=True)
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

def _prepare_eval_data(dataset, edge_map):
    dataset = dataset.sample(frac = .1)
    dataset["time_of_day"] = dataset["time"] / (24*60*60)
    labels = dataset["rel_speed"].copy()

    features = np.zeros((len(dataset), len(edge_map)+1)) #one-hot-encoded edge_id | time_of_day
    i = 0
    for _,r in dataset.iterrows():
        features[i,edge_map[r["edge"]]] = 1.0
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
        
    adversarial_targets = np.random.choice(vehs["first_2h"], 200, replace=False)
    adversarial_global_trainers = np.random.choice(list(set(vehs["train_vehs"]+vehs["test_vehs"]).difference(adversarial_targets)), 3700, replace=False)
    test_vehicles = adversarial_targets

    #pretraining a global data:
    global_pretrained = pretrain_model(edge_map, adversarial_global_trainers)
    
    #prepare baseline for inference methods:
    global_predictions = {}
    #print("Computing reference predictions.", flush=True)
    with tf.device('/CPU:0'):
        for p in edge_map:
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
                for vehicle in test_vehicles[veh_range:(veh_range+VEHICLES_PER_COMM_ROUND)]:
                    address = i%len(addresses)
                    arguments.append({
                        "vehicle": vehicle,
                        "address": addresses[address],
                        "model_weights": weights,
                        "sharing_method": sm,
                        "global_predictions": global_predictions
                    })
                    i += 1

                #print("Dataset prepared. Start training.", flush=True)
                with Pool(NUM_CLIENTS) as pool:
                    results = pool.map(train_client, arguments)
                    for r in results:
                        if ("inference_results" in r):
                            inference_results[r["vehicle"]] = r["inference_results"]


            with open(f"/{RESULT_PATH}/{sm}/inference_results.json", "w") as f:
                json.dump(inference_results, f)

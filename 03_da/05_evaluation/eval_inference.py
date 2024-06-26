import sumolib
import numpy as np
from copy import deepcopy

def calculate_positives_negatives(sent_set: set, path_set: set) -> (int, int):
    positives, negatives = 0, 0
    for se in sent_set:
        if se in path_set:
            positives += 1
        else:
            negatives += 1
    return positives, negatives
    

def evaluate_metrics(data_edges: list, predicted_path: list, true_path: list) -> (float, float):
    tp, fp = 0, 0
    true_path = deepcopy(true_path)
    true_path = set(true_path)
    
    if len(true_path) == 0:
        return -1, -1
    for x in predicted_path:
        if (x in true_path) and (x in data_edges):
            tp += 1
        else:
            fp += 1
    return tp, fp

def evaluate_distance(net, predicted_path: list, true_path: list, idx_to_edge_map: dict) -> float:
    '''Driving distance between predicted and true origin'''
    def get_euclidean_distance(net, edge1, edge2):
        def _get_center_point(net, edge):
            x = (net.getEdge(edge).getBoundingBox()[0] + net.getEdge(edge).getBoundingBox()[2])/2
            y = (net.getEdge(edge).getBoundingBox()[1] + net.getEdge(edge).getBoundingBox()[3])/2
            return x, y
    
        x1, y1 = _get_center_point(net, edge1)
        x2, y2 = _get_center_point(net, edge2)
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    #print(predicted_path, true_path)
    pred_orig = idx_to_edge_map[str(predicted_path[0])]
    true_orig = idx_to_edge_map[str(true_path[0])]
    return get_euclidean_distance(net, pred_orig, true_orig)
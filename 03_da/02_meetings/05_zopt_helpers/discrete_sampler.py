import numpy as np
import pandas as pd

def make_maximal_sample(sample, ideal, ideal_keys, selection = np.ceil, primal_data=None):
    '''
        ideal: shall be sorted!
        sample: indices shall be the same as ideal!
    '''
    scale = 100000
    
    i = 0
    while (i<len(ideal_keys)) and (len(sample[ideal_keys[i]])>0):
        if len(sample[ideal_keys[i]])<np.round(scale*ideal[ideal_keys[i]]):
            scale = len(sample[ideal_keys[i]])/ideal[ideal_keys[i]]
        i += 1

    #handling if the least possible category would contain more than 1 samples:
    if (i<len(ideal_keys)) and (selection(scale*ideal[ideal_keys[i-1]]) > 1) or (
        i>=len(ideal_keys) and (selection(scale*ideal[ideal_keys[-1]]) > 1)):
        
        scale = 1/ideal[ideal_keys[i-1]] if i<len(ideal_keys) else 1/ideal[ideal_keys[-1]]
        
    sampled = {}
    i = 0
    while (i<len(ideal_keys)) and (len(sample[ideal_keys[i]])>0):
        x = ideal_keys[i]
        nsamples = int(selection((scale*ideal[x])))
        sampled[x] = sample[x][:nsamples]
        i += 1
    while (i<len(ideal_keys)):
        sampled[ideal_keys[i]] = []
        i += 1

    if not(primal_data is None):
        for edge in primal_data:
            if edge in sampled:
                sampled[edge] += primal_data[edge]
            else:
                sampled[edge] = primal_data[edge]

    return sampled


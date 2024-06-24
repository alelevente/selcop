import numpy as np
import pandas as pd

TIME_WINDOW = 15*60 #[s]

def create_difference_dataset(global_predict, vehicle_predict):
    '''
        Creates a per parking list of the differences in prediction between the global and
        a vehicle's model.
        
        Parameters:
            - global_predict: prediction of the global model per parking lots
            - vehicle_predict: prediction of a local vehicle per parking lots
            
        Returns:
            - prediction differences
    '''
    p_diffs = {}

    for p in global_predict:
        pred_vehicle = np.array(vehicle_predict[p])
        pred_global = global_predict[p]
        
        p_diffs[p] = (pred_vehicle-pred_global)**2
                 
    return p_diffs

def predict_eval_positions(p_diffs, true_parkings, num_parking_lots = 10):
    '''
        Maliciously infers possibly visited parking lots from
        prediction differences. Then evaluates the success rate of the attacker.
        
        Parameters:
            - p_diffs: prediction differences
            - true_parkings: list of the parking lots which were visited by a vehicle
            - num_parking_lots: how many lots try to guess
            
        Returns:
            - the successfully identified parking lots (out of the prescribed num_parking_lots)
    '''
    
    p_diff_means = {}
    for p in p_diffs:
        p_diff_means[p] = np.mean(p_diffs[p])

    p_diff_series = pd.Series(p_diff_means)

    #converting to sets to be able to get the prediction as an intersection
    predicted_ps = set(p_diff_series.nlargest(num_parking_lots).index)
    true_parkings = set(true_parkings)
    intersection = predicted_ps.intersection(true_parkings)
    return intersection

def search_nearest_move_bin_(pred_time, move_bins_bins, move_bins_counts):
    '''
        From a moving time histogram and a prediction time, it calculates the
        offset between a true moving time and the prediction time.
        
        Parameters:
            - pred_time: predicted moving time in [0,1) range
            - move_bins_bins: x values of the histogram (bin ranges)
            - move_bins_counts: y values of the histogram
            
        Returns:
            - an offset, if negative: the predicted value is later than the closest true moving
    '''
    
    ref_idx, = np.where(move_bins_bins == pred_time)[0]
    rel_idx = 0
    #print(f"ref_idx: {ref_idx}\nlenght: {len(move_bins_counts)}")
    #searching backward:
    while (ref_idx+rel_idx >= 0) and (move_bins_counts[ref_idx+rel_idx]==0):
        rel_idx -= 1
    down_step = 1
    if rel_idx<=0: #found some data
        down_step = rel_idx
    rel_idx = 0
    #searching forward:
    while (ref_idx+rel_idx < len(move_bins_counts)) and (move_bins_counts[ref_idx+rel_idx]==0):
        rel_idx += 1
    if rel_idx < (24*60*60)//TIME_WINDOW: #found some data
        if down_step == 1:
            return rel_idx
        else:
            return down_step if abs(down_step)<rel_idx else rel_idx
    else:
        return None if down_step == 1 else down_step

def predict_eval_time(p_diffs, true_moving_times, time_window=900):
    '''
        Maliciously infers possible moving time from
        prediction differences. Then evalutes how many time windows
        the prediction is away from a true moving of the vehicle.
        
        Parameters:
            - p_diffs: prediction_differences
            - true_moving_times: data series describing the true moving times
            - time_window: how long is a time window in seconds
            
        Returns:
            - an offset, how many time window is away the best prediction from
              a true moving of the vehicle
    '''
    
    time_diffs = {}
    for p in p_diffs:
        for t in range(0, 24*60*60, time_window):
            if t in time_diffs:
                time_diffs[t] += np.mean(p_diffs[p][t:t+time_window])
            else:
                time_diffs[t] = np.mean(p_diffs[p][t:t+time_window])
    
    time_diffs_series = pd.Series(time_diffs) #to be able to run handy functions
    
    prediction_diff_rates_x = np.arange(time_window, 24*60*60, time_window) #1 step shorter because of the differentiation
    prediction_diff_rates_y = np.abs(np.diff(time_diffs_series.values)) #|d/dt(time_diff(x, t))|
    prediction_diff_rates = pd.Series(data = prediction_diff_rates_y, index = prediction_diff_rates_x)
    pred_time = np.clip(prediction_diff_rates.index[prediction_diff_rates.argmax()]/(24*60*60), a_min=0.0, a_max=.9999999)
    #creating the histogram:
    move_bins_counts, move_bins_bins = np.histogram(true_moving_times, bins=np.arange(0, 24*60*60+1, time_window)/(24*60*60))
    #calculating the offset:
    #print(f"predicted_time: {pred_time}")
    #print(f"move_bins_bins: {move_bins_bins}")
    offset = search_nearest_move_bin_(pred_time, move_bins_bins, move_bins_counts)
    return offset
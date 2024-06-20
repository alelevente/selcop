'''
    FUNCTIONS FOR RUNNING A ZEUTHEN INSPIRED OPTIMIZATION STRATEGY
'''

import numpy as np
import pandas as pd
import copy

import discrete_sampler

import matplotlib.pyplot as plt

from importlib import reload
reload(discrete_sampler)

FALL_BACK_TIME = 15 #[s]

###############################################################################
##################### METHODS FOR RISK CALCULATION ############################
def calculate_num_shared(dataset):
    n_samples = 0
    for edge in dataset:
        n_samples += len(dataset[edge])
    return n_samples

def alter_utility(dataset, EPA):
    n_samples = calculate_num_shared(dataset)
    if n_samples == 0:
        return 0
    return  1+1/EPA-1/(n_samples)

def ego_utility(dataset, EPA):
    n_samples = calculate_num_shared(dataset)
    return 1+1/EPA-n_samples/EPA

def calc_alter_risk(alters_concession, egos_concession, EPA, verbose=False):
    ua_da = alter_utility(alters_concession, EPA)
    ua_de = alter_utility(egos_concession, EPA)
    if verbose:
        print("ua_de: %.4f, ua_da: %.4f"%(ua_de, ua_da))
    return (ua_da-ua_de)/ua_da

def calc_ego_risk(egos_concession, alters_concession, EPA, verbose=False):
    ue_de = ego_utility(egos_concession, EPA)
    ue_da = ego_utility(alters_concession, EPA)
    if verbose:
        print("ue_de: %.4f, ue_da: %.4f"%(ue_de, ue_da))
    return (ue_de-ue_da)/ue_de

def calc_ideal(EPA):
    return np.sqrt(EPA)
###############################################################################

###############################################################################
######################### METHODS FOR DATA HANDLING ###########################

def prepare_dataset(dataset, probabilities, idx_to_edge):
    probs_arg = np.argsort(probabilities)
    
    to_sample = {}
    probs = {}
    idx_list = []

    for i in range(len(probs_arg)-1, -1, -1):
        idx = str(probs_arg[i])
        edge = idx_to_edge[idx]

        edge_data = dataset[dataset["edge"] == edge]

        alters_data = edge_data[edge_data["receive_time"]>-1]
        alters_data = alters_data.sort_values(by=["time"])
        ego_data = edge_data[edge_data["receive_time"] == -1]
        ego_data = ego_data.sort_values(by=["time"])
        edge_df = pd.concat([alters_data, ego_data])

        #it is easier to handle the data in list than in a pd.DataFrame:
        edge_data = []
        for _,r in edge_df.iterrows():
            edge_data.append(r.to_dict())

        to_sample[edge] = edge_data
        probs[edge] = probabilities[probs_arg[i]]
        idx_list.append(edge)

    return to_sample, probs, idx_list


def create_data_to_send(dataset):
    send_data = None
    for edge in dataset:
        if send_data is None:
            send_data = pd.DataFrame.from_records(dataset[edge])
        else:
            send_data = pd.concat([send_data, pd.DataFrame.from_records(dataset[edge])], ignore_index=True)
    return send_data
###############################################################################

###############################################################################
##################### METHODS FOR CONCESSION MAKING ###########################
def new_ego_concession(last_deal, not_yet_in_deal, possible_steps,
                       ideal, ideal_keys, last_added_idx, EPA,
                       primal_data = None,
                       verbose=False):
    assert not(last_deal is None), "last_deal shall not be None"
    
    last_deal = copy.deepcopy(last_deal)
    for edge in not_yet_in_deal:
        for x in not_yet_in_deal[edge]:
            last_deal[edge].append(x)
        
    if verbose:
        print("ego's last_deal:", last_deal)
    ####################### HANDLING POSSIBLE NEXT EDGES ####################################
    change_on = None
    last_util = ego_utility(last_deal, EPA)
    
    min_util_loss = 10
    min_step = None
    if len(possible_steps)>0:
        for edge in possible_steps:
            if len(possible_steps[edge]) > 0:
                calc_deal = copy.deepcopy(last_deal)
                if not edge in calc_deal:
                    calc_deal[edge] = []
                calc_deal[edge].append(possible_steps[edge][0])
                possible_deal = discrete_sampler.make_maximal_sample(calc_deal, ideal, ideal_keys)
                if verbose:
                    print("ego's possible_deal: ",possible_deal)
                util = ego_utility(possible_deal, EPA)
                util_change = last_util - util
                if verbose:
                    print("ego's last_util: %.4f \t util: %.4f \t util_change: %.4f"%(last_util, util, util_change))        
                if (util_change < min_util_loss) and (util_change > 0):
                    min_util_loss = last_util - util
                    min_step = possible_deal
                    change_on = edge
            
        if change_on is None: # if we cannot change the resampled list
            #then we add a new edge following a round robin sampling:
            while (last_added_idx<len(ideal)) and (not(ideal_keys[last_added_idx] in possible_steps)): last_added_idx += 1
            if last_added_idx<len(ideal):
                if not(ideal_keys[last_added_idx] in not_yet_in_deal):
                    not_yet_in_deal[ideal_keys[last_added_idx]] = []
                not_yet_in_deal[ideal_keys[last_added_idx]].append(possible_steps[ideal_keys[last_added_idx]][0])
                del possible_steps[ideal_keys[last_added_idx]][0]
                if len(possible_steps[ideal_keys[last_added_idx]]) == 0:
                    del possible_steps[ideal_keys[last_added_idx]]
            min_step = None
        else:
            del possible_steps[change_on][0]
            if len(possible_steps[change_on]) == 0:
                    del possible_steps[change_on]
            not_yet_in_deal_update = copy.deepcopy(not_yet_in_deal)
            #handling already selected but not sampled data:
            for edge in min_step:
                for x in edge:
                    if edge in not_yet_in_deal:
                        if x in not_yet_in_deal_update[edge]:
                            not_yet_in_deal_update[edge].remove(x)
            not_yet_in_deal = not_yet_in_deal_update
            last_added_idx = -1 #resetting the round robin counter
    ###################### LAST DEAL #######################
    else:
        calc_deal = copy.deepcopy(last_deal)
        if verbose:
            print("ego's last calc_deal:", calc_deal)
        min_step = discrete_sampler.make_maximal_sample(calc_deal, ideal, ideal_keys)
        not_yet_in_deal = {} #everything is added, no further steps are possible

    return min_step, not_yet_in_deal, possible_steps, last_added_idx

def new_alter_concession(last_deal, not_yet_in_deal, possible_steps, ideal, ideal_keys, EPA, verbose=False):
    assert not(last_deal is None), "last_deal shall not be None"
    if len(possible_steps) == 0:
        return None, not_yet_in_deal, possible_steps
    
    last_util = alter_utility(last_deal, EPA)
    
    min_util_loss = 10
    min_step = None
    last_deal = copy.deepcopy(last_deal)
    for edge in not_yet_in_deal:
        for x in not_yet_in_deal[edge]:
            last_deal[edge].remove(x)
        
    if verbose:
        print("alter's last_deal:", last_deal)
    for edge in possible_steps:
        if len(last_deal[edge]) > 0:
            calc_deal = copy.deepcopy(last_deal)
            if verbose:
                print("alter's removing: ", edge)
                print("alter's possible steps: ", possible_steps)
            calc_deal[edge].remove(possible_steps[edge][-1])
            possible_deal = discrete_sampler.make_maximal_sample(calc_deal, ideal, ideal_keys)
            if verbose:
                print("alter's new possible_deal: ",possible_deal)
            util = alter_utility(possible_deal, EPA)
            util_change = last_util - util
            if verbose:
                print("Alter's last_util: %.4f \t util: %.4f \t util_change: %.4f"%(last_util, util, util_change))        
            if (util_change < min_util_loss) and (util_change > 0):
                min_util_loss = last_util - util
                min_step = possible_deal
                change_on = edge
            
    to_remove = []
    #searching for elements to left out:
    for edge in possible_steps:
        for x in possible_steps[edge]:
            if not(x in min_step[edge]):
                to_remove.append((edge, x))
    for edge, x in to_remove:
        possible_steps[edge].remove(x)
    #searching for edges to remove:
    to_remove_edges = []
    for edge in possible_steps:
        if len(possible_steps[edge]) == 0:
            to_remove_edges.append(edge)    
    for edge in to_remove_edges:
        del possible_steps[edge]
    
    not_yet_in_deal = {}
    return min_step, not_yet_in_deal, possible_steps

def first_ego_concession(ego_data, ideal, ideal_keys):
    ego_others_data = {}
    for edge in ego_data:
        ego_others_data[edge] = []
        for x in ego_data[edge]:
            if x["receive_time"]>-1:
                ego_others_data[edge].append(x)
    max_sample = discrete_sampler.make_maximal_sample(ego_others_data, ideal, ideal_keys)
    return max_sample

def first_alter_concession(dataset, ideal, ideal_keys):
    return discrete_sampler.make_maximal_sample(dataset, ideal, ideal_keys)

def calculate_differences(bigger_set, smaller_set):
    difference_set = {}
    for edge in bigger_set:
        diff = []
        if edge in smaller_set:
            for deb in bigger_set[edge]:
                smaller_idx = 0
                #searching for element in bigger set in the smaller set:
                while (smaller_idx < len(smaller_set[edge])) and (smaller_set[edge][smaller_idx]["hash"] != deb["hash"]): smaller_idx += 1
                if smaller_idx >= len(smaller_set[edge]): # if not found
                    diff.append(deb)
        else:
            diff = bigger_set[edge]
        if len(diff) > 0:
            difference_set[edge] = diff
    return difference_set
###############################################################################

###############################################################################
##################### METHODS OF THE ZOPT STRATEGY ############################
def run_zopt(dataset, probabilities, idx_to_edge, position, time, verbose=False):
    proc_dataset, ideal, ideal_keys = prepare_dataset(dataset, probabilities, idx_to_edge)
    first_ego = first_ego_concession(proc_dataset, ideal, ideal_keys)
    first_alter = first_alter_concession(proc_dataset, ideal, ideal_keys)
    diff_alter_and_ego = calculate_differences(first_alter, first_ego)
    EPA = calculate_num_shared(first_alter)

    if (EPA == 0) or (len(dataset) == 0):
        return {}, ([],[],[],[], -1, -1, True)

    #fallback strategy in case ego has no data:
    '''primal_data = None
    if calculate_num_shared(first_ego) == 0:
        last_data = dataset[dataset["time"] >= time-FALL_BACK_TIME]
        primal_data,_,_ = prepare_dataset(last_data, probabilities, idx_to_edge)
        if calculate_num_shared(primal_data)>0:
            return primal_data, ([],[],[],[], -1, -1, True)
        else:
            return {}, ([],[],[],[], -1, -1, False)'''

    if verbose:
        print("========================================")
        print(f"EPA:\t{EPA}")
        print("First deals:")
        print("Ego: ", first_ego, "\nAlter: ", first_alter)
        print("Difference: ", diff_alter_and_ego)

    act_deal_ego = first_ego
    not_yet_in_ego = {}
    possible_moves_ego = copy.deepcopy(diff_alter_and_ego)
    act_deal_alter = first_alter
    not_yet_in_alter = {}
    possible_moves_alter = copy.deepcopy(diff_alter_and_ego)
    
    ue_des, ue_das, ua_des, ua_das = [], [], [], []
    ue_des.append(ego_utility(act_deal_ego, EPA))
    ue_das.append(ego_utility(act_deal_alter, EPA))
    ua_des.append(alter_utility(act_deal_ego, EPA))
    ua_das.append(alter_utility(act_deal_alter, EPA))
    last_added_idx = 0
    while ego_utility(act_deal_alter, EPA) < ego_utility(act_deal_ego, EPA):
        ego_risk = calc_ego_risk(act_deal_ego, act_deal_alter, EPA, verbose)
        alter_risk = calc_alter_risk(act_deal_alter, act_deal_ego, EPA, verbose)
        if verbose:
            print("============================")
            print("Alter's risk:\t%.4f\t\t Ego's risk:\t%.4f"%(alter_risk, ego_risk))
        if ego_risk < alter_risk:
            if verbose:
                print("Ego creates a new concession")
            deal_ego, not_yet_in_ego, possible_moves_ego, last_added_idx = new_ego_concession(act_deal_ego, not_yet_in_ego, possible_moves_ego,
                                                                              ideal, ideal_keys,
                                                                              last_added_idx, EPA,
                                                                              primal_data=None,
                                                                              verbose=verbose)
            act_deal_ego = deal_ego if not (deal_ego is None) else act_deal_ego
        else:
            if verbose:
                print("Alter creates a new concession")
            deal_alter, not_yet_in_alter, possible_moves_alter = new_alter_concession(act_deal_alter, not_yet_in_alter,
                                                                                          possible_moves_alter,
                                                                                          ideal, ideal_keys,
                                                                                          EPA, verbose)
            act_deal_alter = deal_alter if not (deal_alter is None) else act_deal_alter
        if verbose:
            print("utility of ego's concession:\t%.4f \t\t alter's concession:\t%.4f"%(ego_utility(act_deal_ego, EPA), ego_utility(act_deal_alter, EPA)))
            print("ue(de)*ua(de):\t%.4f \t ue(da)*ua(da)\t%.4f"%(ego_utility(act_deal_ego, EPA)*alter_utility(act_deal_ego, EPA),
                                                                (ego_utility(act_deal_alter, EPA)*alter_utility(act_deal_alter, EPA))))
        ue_des.append(ego_utility(act_deal_ego, EPA))
        ue_das.append(ego_utility(act_deal_alter, EPA))
        ua_des.append(alter_utility(act_deal_ego, EPA))
        ua_das.append(alter_utility(act_deal_alter, EPA))
        
    ideal = calc_ideal(EPA)
    factual = calculate_num_shared(act_deal_alter)
    return act_deal_alter, (ue_des, ue_das, ua_des, ua_das, ideal, factual, False)
###############################################################################

###############################################################################
################### CALCULATING SENDING DATA ##################################

def calc_send_data(dataset, probabilities, idx_to_edge, position, time, verbose=False):
    send_data, performance = run_zopt(dataset, probabilities, idx_to_edge, position, time, verbose)
    perf_dict = {
        "ego_utilities_ego_deals": performance[0],
        "ego_utilities_alter_deals": performance[1],
        "alter_utilities_ego_deals": performance[2],
        "alter_utilities_alter_deals": performance[3],
        "ideal_datasize": performance[4],
        "factual_datasize": performance[5],
        "fall_back": performance[6]
    }
    send_data = create_data_to_send(send_data)
    return send_data, perf_dict

###############################################################################

###############################################################################
###################### VISUALIZATION OF RESULTS ###############################
def plot_utilities(utilities):
    ue_des, ue_das, ua_des, ua_das = utilities
    fig, ax = plt.subplots()
    ax.scatter(ue_des, ua_des, alpha=0.4, label="ego's concessions")
    ax.scatter(ue_das, ua_das, alpha=0.4, label="alter's concessions")
    ax.scatter(ue_das[-1:], ua_das[-1:], marker="x", s=75, c="red", label="final")
    ax.set_xlabel("Ego's utility")
    ax.set_ylabel("Alter's utility")
    ax.set_xlim(0)
    ax.set_ylim(0)
    return fig, ax
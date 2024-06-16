import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
import pandas as pd

import json

import sys,os

SUMO_HOME = os.environ["SUMO_HOME"] #locating the simulator
sys.path.append(SUMO_HOME+"/tools")
import sumolib
from sumolib.visualization import helpers


class Option:
    #default options required by sumolib.visualization
    defaultWidth = 2
    defaultColor = (0.0, 0.0, 0.0, 0.0)
    linestyle = "solid"
    
    
def plot_dataset(net_file, vehicle_parkings, parking_position, title="",
                       color=(0.7, 0.0, 0.7, 0.66),
                       fig=None, ax=None):

    '''
        Plots a road network with edges colored according to a probabilistic distribution.
        Parameters:
            net_file: path to the net file
            vehicle_parking: list of parking lots measured by the vehicle
                If an edge is not in this map, it will get a default (light gray) color.
            parking_position:
                A dictionary of which keys are the parking lot names, and the values are the
                edge on which the parking lot resides.
            title: title of the produced plot
            color: color of the edges
            fig: if None then a new map is created; if it is given, then only special edges are overplot to the original fig
            ax: see fig

        Returns:
            a figure and axis object
    '''
    
    net = sumolib.net.readNet(net_file)
        
    scalar_map = None
    colors = {}
    options = Option()
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(22, 20))
        for e in parking_position.values():
            colors[e] = np.array((0.125, 0.125, 0.125, .25)) #edges are gray by default
            
    for pl in vehicle_parkings:
        edge = parking_position[pl]
        colors[edge] = color

    helpers.plotNet(net, colors, [], options)
    plt.title(title)
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    ax.set_facecolor("lightgray")
    if not(scalar_map is None):
        plt.colorbar(scalar_map)

    return fig, ax

def plot_network_probs(net_file, probabilities, index_to_edge_map, cmap="YlGn",
                       title="", special_edges=None,
                       special_color=(1.0, 0.0, 0.0, 1.0),
                       fig=None, ax=None):

    '''
        Plots a road network with edges colored according to a probabilistic distribution.
        Parameters:
            net_file: a sumo road network file path
            probabilities: a dictionary that maps edge indices to probabilities
                If an edge is not in this map, it will get a default (light gray) color.
            index_to_edge_map: a dictionary that maps edge indices to SUMO edge IDs
            cmap: the colormap to be used on the plot
            title: title of the produced plot
            special_edges: edges to be plotted with special color, given in a similar structure
                as probabilities parameter
            special_color: color of the special edges (RGBA)
            fig: if None then a new map is created; if it is given, then only special edges are overplot to the original fig
            ax: see fig

        Returns:
            a figure and axis object
    '''

    net = sumolib.net.readNet(net_file)
        
    scalar_map = None
    colors = {}
    options = Option()
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(22, 20))
        if probabilities is None:
            for e in index_to_edge_map.values():
                colors[e] = (0.125, 0.125, 0.125, .25)
        else:
            c_norm = matplotlib.colors.Normalize(vmin=min(probabilities)*0.85, vmax=max(probabilities)*1.15)
            scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
            for i,p in enumerate(probabilities):
                if (p == 0.0) and (str(i) in index_to_edge_map):
                    colors[index_to_edge_map[str(i)]] = (0.125, 0.125, .125, .125)
                elif str(i) in index_to_edge_map:
                    colors[index_to_edge_map[str(i)]] = scalar_map.to_rgba(min(1,max(0,p)))
                    
    if not(special_edges is None):
        for ind in special_edges:
            colors[index_to_edge_map[ind]] = special_color
    
    helpers.plotNet(net, colors, [], options)
    plt.title(title)
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    ax.set_facecolor("lightgray")
    if not(scalar_map is None):
        plt.colorbar(scalar_map)

    return fig, ax


def plot_dataset(net_file, own_edges, others_edges, vehicle_route, index_to_edge_map, title="",
                       own_color=(0.0, 1.0, 0.0, 0.25),
                       others_color=(0.0, 0.0, 1.0, 0.25),
                       special_edges=None,
                       special_color=(1.0, 0.0, 0.0, 0.25),
                       fig=None, ax=None):

    '''
        Plots a road network with edges colored according to a probabilistic distribution.
        Parameters:
            net: a sumolib road network
            own_edges: set of own edges
            others_edges: set of others' edges
            vehicle_route: set of the edges along the road of the vehicle
            index_to_edge_map: a dictionary that maps edge indices to SUMO edge IDs
            title: title of the produced plot
            own_color: color for the is_own=True edges
            others_color: color for the is_own=False edges
            fig: if None then a new map is created; if it is given, then only special edges are overplot to the original fig
            ax: see fig

        Returns:
            a figure and axis object
    '''
    
    net = sumolib.net.readNet(net_file)
        
    scalar_map = None
    colors = {}
    options = Option()
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(22, 20))
        for e in index_to_edge_map.values():
            colors[e] = np.array((0.125, 0.125, 0.125, .25))
                        
    for edge in vehicle_route:
        colors[edge] += np.array((0.0, 1.0, 0.0, 0.25))
    for edge in own_edges:
        colors[edge] += np.array(own_color)
    for edge in others_edges:
        colors[edge] += np.array(others_color)
    if not(special_edges is None):
        for ind in special_edges:
            colors[ind] += np.array(special_color)
    for c in colors:
        #colors[c] = colors[c]/np.sum(colors[c])
        colors[c] = np.clip(colors[c], a_min=0.0, a_max=1.0)
    helpers.plotNet(net, colors, [], options)
    plt.title(title)
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    ax.set_facecolor("lightgray")
    if not(scalar_map is None):
        plt.colorbar(scalar_map)

    return fig, ax
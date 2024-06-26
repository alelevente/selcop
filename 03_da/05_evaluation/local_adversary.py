'''
    FUNCTIONS FOR INFERING PRIVATE LOCATION DATA BASED ON LOCALLY EXCHANGED DATA.
'''
import numpy as np
import copy

MAX_DEPTH = 20

######################## DATA STRUCTURE FOR INFERENCE #########################
class DataTree:
    '''
        BFS-tree built from vehicle data. BFS represents the vehicles' path
        efficiently: each branch corresponds to a specific vehicle.
    '''
    class Node:
        def __init__(self, idx):
            self.children = []
            self.parent = None
            self.idx = idx
            
    def dfs_build(root, points, tcs):
        my_node = DataTree.Node(root)
        if not(root in points):
            return None
        points = copy.deepcopy(points)
        points.remove(root)
        i = 0
        while i<len(points):
            if tcs[root][i]>0:
                node, points = DataTree.bfs_build(i, points, tcs)
                my_node.children.append(node)
            else:
                i += 1
        return my_node, points
    
    def bfs_build(root, points, tcs):
        explored = [root]
        remaining = [root]
        nodes = {}
        root_node = DataTree.Node(root)
        nodes[root] = root_node
        
        while len(remaining)>0:
            edge = remaining[0]
            del remaining[0]
            explored.append(edge)
            for i in points: 
                if (tcs[edge][i] > 0) and (not i in explored): #adjacency
                    edge_node = DataTree.Node(i)
                    nodes[i] = edge_node
                    edge_node.parent = nodes[edge]
                    nodes[edge].children.append(edge_node)
                    remaining.append(i)
        return root_node, nodes
    
    def __str__(self):
        answer = ""
        for n in self.nodes:
            node = self.nodes[n]
            answer = answer + str(n)+": ["
            children_indices = []
            for ch in node.children:
                answer = answer + str(ch.idx)+", "
            answer = answer + "] \n"
        return answer
                    
    def _depth_calc(node, depth=0):
        answer = {node.idx: depth}
        for c in node.children:
            answ_dict = DataTree._depth_calc(c, depth+1)
            answer.update(answ_dict)
        return answer
    
    
    def list_elements(tree):
        elements = [tree.idx]
        for c in tree.children:
            #print(c.idx)
            elements += DataTree.list_elements(c)
        return elements
    
    def __init__(self, root, points, tcs):
        self.root, self.nodes = DataTree.bfs_build(root, points, tcs)
        self.depth_dict = DataTree._depth_calc(self.root)
        
    def get_branching_factor(node):
        if node.children == []:
            return 1
        
        branching_factor = 0
        for ch in node.children:
            branching_factor += DataTree.get_branching_factor(ch)
        return branching_factor
    
    def get_path_from_origin(self, origin):
        path = [origin]
        act = self.nodes[origin].idx
        while not(act is None):
            parent = self.nodes[act].parent
            if not(parent is None):
                act = parent.idx
                path.append(act)
            else:
                act = None
        return path       
    
######################## INFERENCE ALGORITHMS #################################
def river_algorithm(data_tree):
    def _get_tallest_subtree(root, current_height = 0):
        if root.children == []:
            return current_height, [root.idx]
        heights, trees = [], []
        for c in root.children:
            h, t = _get_tallest_subtree(c, current_height+1)
            heights.append(h)
            t.append(root.idx)
            trees.append(t)
        tallest_subtree_elements = trees[np.argmax(heights)]
        return max(heights), tallest_subtree_elements
    
    h, tr = _get_tallest_subtree(data_tree.root)
    return h, tr

def min_prior_algorithm(data_tree, probabilities):
    if len(probabilities) == 0: return [data_tree.root.idx]
    min_edge_idx = list(probabilities.keys())[0]
    for edge_idx in probabilities:
        if probabilities[min_edge_idx] > probabilities[edge_idx]:
            min_edge_idx = edge_idx
    path = DataTree.get_path_from_origin(data_tree, min_edge_idx)
    return path

def time_bounded_river_algorithm(data_tree, time_labels, current_time):
    def _get_tallest_subtree(root, current_time, current_height = 0):
        if root.children == []:
            return current_height, [root.idx]
        heights, trees = [], []
        for c in root.children:
            #print(current_time, time_labels[c.idx])
            if time_labels[c.idx]<current_time:
                h, t = _get_tallest_subtree(c,
                                            time_labels[c.idx],
                                            current_height+1)
                heights.append(h)
                t.append(root.idx)
                trees.append(t)
        if heights==[]:
            return current_height, [root.idx]
        tallest_subtree_elements = trees[np.argmax(heights)]
        return max(heights), tallest_subtree_elements
    
    if not(data_tree.root.idx in time_labels):
        time_labels[data_tree.root.idx] = current_time
    h, tr = _get_tallest_subtree(data_tree.root,
                                 time_labels[data_tree.root.idx])
    return h, tr

def calculate_probabilities_for_minp(data_tree, backward_probs_per_step, actual_position_idx):
    bps = backward_probs_per_step[actual_position_idx]
    answer = {}
    for edge_idx in data_tree.depth_dict:
        if (data_tree.depth_dict[edge_idx] != 0) and (
            data_tree.depth_dict[edge_idx] <= MAX_DEPTH):
            answer[edge_idx] = bps[data_tree.depth_dict[edge_idx]-1][edge_idx]
            #print(edge_idx, bps[data_tree.depth_dict[edge_idx]-1])
#    print(data_tree.depth_dict, answer)
    return answer
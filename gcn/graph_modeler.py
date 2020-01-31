### Graph Modeler
import os
from pathlib import Path
import json
from collections import namedtuple
import csv
import cv2 
import numpy as np

from data_access import *
from visuals import *

# ------------------------------------------------------------------------------------------
# declarations
UNDEFINED="undefined"
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')
cv_size = lambda img: tuple(img.shape[1::-1])

#### Line Formation (Algorithm 1)
# ------------------------------------------------------------------------------------------
def is_in_line(wa, wb):
    """
        Determine if two words belong in the same horizontal line
        In an image, the y coordinate runs from top to bottom so wa.top < wa.bottom
    """
    return wa.top < wb.bottom and wa.bottom > wb.top

        
def line_formation(word_areas):
    """
    Group the words into horizontal lines (algorithm 1)
    The words in the output have been replaced by their positional index in the word_boxes list.
    Output
        lines: (list) a list of lines sorted from top to bottom, by the top y of the first word. 
                Each line is a list of words sorted from left to right by ther left x.
        
    """
    lines, line = [], []
    
    # before processing, word areas are sorted from top to bottom on their top y coordinate (top)
    word_areas = sorted(word_areas, key=lambda wa: wa.top)

    # The first line is initialized with the fist word as head
    # then we start processing with the second word
    line = [word_areas[0]]
    for word_area in word_areas[1:]:
        #  check if the word_area is aligned with the first word of the current line
        if is_in_line(line[0], word_area):
            line.append(word_area)
        else:
            # words in line are sorted from left to right by their left x coordinate (left)
            lines.append(sorted(line, key=lambda wa: wa.left))
            # a new line is initialized with the current word as head
            line = [word_area]

    # append the last line
    lines.append(sorted(line, key=lambda bw: bw[0]))

    return lines


#### Graph Modeling (Algorithm 2)
# ------------------------------------------------------------------------------------------
def get_left_edge(word_pos, line, word_areas, image_width):
    """
    Return a tuple with the left neighbour and the relative distance.
    The relative distance to the left neighbour is negative
    """
    # if the source_word is the first word of the line, then return an empty edge
    if word_pos == 0:
        left_neighbour_idx = UNDEFINED
        left_neighbour_relative_distance = UNDEFINED
    else:
        source_word = line[word_pos]
        left_neighbour_idx = line[word_pos-1].idx
        left_neighbour_relative_distance = round((line[word_pos-1].right - source_word.left)/image_width, 3)

    return (left_neighbour_idx, left_neighbour_relative_distance)
    
def get_right_edge(word_pos, line, word_areas, image_width):
    """
    Return a tuple with the right neighbour and the relative distance.
    The relative distance to the right neighbour is positive
    """
    # if the source_word is the last word of the line, then return an empty edge
    if word_pos == len(line)-1:
        right_neighbour_idx = UNDEFINED
        right_neighbour_relative_distance = UNDEFINED
    else:
        source_word = line[word_pos]
        right_neighbour_idx = line[word_pos+1].idx
        right_neighbour_relative_distance = round((line[word_pos+1].left - source_word.right)/image_width, 3)

    return (right_neighbour_idx, right_neighbour_relative_distance)

# ------------------------------------------------------------------------------------------
def is_in_column(wa, wb):
    """
        Determine if two words belong in the same vertical column 
        In an image, the x coordinate runs from left to right so wa.left < wa.right
    """
    return wa.left < wb.right and wa.right > wb.left

def form_column(source_word, word_areas):
    """
    returns a list containing all the words that overlap with the 
    source word in the vertical direction and are above the source word
    In an image, the x coordinate runs from left to right so wa.left < wa.right
    """ 
    column = []
    for wa in word_areas:
        # skip words that are not above the source word (skips the source word itself)
        if wa.bottom > source_word.top:
            continue
        if is_in_column(source_word, wa):
            column.append(wa)
    return column
    
def  assign_top_neighbour(source_word, graph, word_areas, image_height):
    """
        Find the words at the minimal vertical distance
        skip words that already have other vertical neighbour 
        In an image, the y coordinate runs from top to bottom so wa_top < wa_bottom
        The relative distance to the top neighbour is negative
        The relative distance to the bottom neighbour is positive
    """ 
    column = form_column(source_word, word_areas)
    min_distance = image_height
    top_neighbour_idx = UNDEFINED
    
    for wa in column:
        
        # skip those words already has a bottom neighbour
        if graph[wa.idx]['bottom'][0] != UNDEFINED:
            continue

        # all the words in column are babove the source word, 
        # so this distance is negative, so we take the absolute value
        top_neighbour_distance = abs(wa.bottom - source_word.top)
        
        if top_neighbour_distance < min_distance:
            min_distance = top_neighbour_distance
            top_neighbour_idx = wa.idx
    
    if top_neighbour_idx != UNDEFINED:
        relative_distance = round(min_distance/image_height, 3)
        graph[source_word.idx]['top'] = (top_neighbour_idx, -relative_distance)
        graph[top_neighbour_idx]['bottom'] = (source_word.idx, relative_distance)


def graph_modeling(lines, word_areas, image_width, image_height):
    """
    Find nearest neighbours for the each word in the document (Algorithm 2)
    input:
        lines: (list) each element is a line, these lines are sorted by vertical position from top to bottom.
        Each line is a list of WordArea tuples sorted by position from left to right.
    output:
        graph: (dict) where each (key, value) item is a node representing a word area,
                the key is an index to the wordArea tuple in the word_areas list, and the value 
                is a dictionary of edges, as follows:
                edges: (dict) where each (key, value) item is an edge to a nearest neighbour
                the key is the direction ('left', 'right', 'top', 'bottom') and the value is 
                a tuple (word_area.idx, relative_distance), for example: ('left': (5, 0.1))
                where 5 is the index of the left neighbour word area in the WordAreas list
    """
    graph ={}
    
    for line in lines:
        for word_pos, source_word in enumerate(line):
            edges = {}
            # neighbours in horizontal direction
            edges['left'] = get_left_edge(word_pos, line,  word_areas, image_width)
            edges['right'] = get_right_edge(word_pos, line, word_areas, image_width)
            edges['top'] = (UNDEFINED, UNDEFINED)
            edges['bottom'] = (UNDEFINED, UNDEFINED)
            graph[source_word.idx] = edges

    for line in lines:
        for word_pos, source_word in enumerate(line):
            # neighbours in vertical direction
            assign_top_neighbour(source_word, graph, word_areas, image_height)           
        
    return graph

#### Adjacency Matrix
def form_adjacency_matrix(graph):
    N = len(graph)
    adj_matrix = np.zeros((N, N))
    for (node_idx, edges) in graph.items():
        for (neighbour_idx, rd) in edges.values():
            if neighbour_idx != UNDEFINED:
                adj_matrix[node_idx, neighbour_idx] = 1
            
    return adj_matrix


# -------------------------------------------------------------------------
# unit tests
# -------------------------------------------------------------------------
def unit_test():
    # configs for normalized data
    PROJECT_ROOT_PREFIX = "/home/adrian/as/blogs/nanonets"
    NORMALIZED_PREFIX = "invoice-ie-with-gcn/data/normalized/"
    normalized_dir = os.path.join(PROJECT_ROOT_PREFIX, NORMALIZED_PREFIX)

     # find files
    image_files, word_files, entity_files = get_normalized_filepaths(normalized_dir)
    # load one raw example
    image, word_areas, entities = load_normalized_example(image_files[0], word_files[0], entity_files[0])

    #test algo 1
    lines = line_formation(word_areas)
    
    # show lines
    for line in lines:
        print('\n')
        for tup in line:
            print(tup)
    
    # test algo 2
    image_width, image_height = cv_size(image)
    graph = graph_modeling(lines, word_areas, image_width, image_height)
    
    # plot image with annotations
    plot_image(image, word_areas, entities, graph, False)    
    print("Visualize Graph Data")
    print("writing {}".format("graph_example.png"))
    plt.savefig('graph_example.png')
    
    # adj matrix
    adjacency_matrix = form_adjacency_matrix(graph)
    print(adjacency_matrix.shape)
    print(np.sum(adjacency_matrix, axis=0))
    print(np.sum(adjacency_matrix, axis=1))
    print(np.sum(adjacency_matrix, axis=0) == np.sum(adjacency_matrix, axis=1))

    print(adjacency_matrix)
    

if __name__ == "__main__":
    unit_test()
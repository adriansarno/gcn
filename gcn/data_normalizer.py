# data_normalizer.py

# pip install editdistance
import os
from pathlib import Path
import json
import editdistance

from collections import namedtuple
import csv
import cv2 

import logging
# ----------------------------------------------------------------------------

# declarations common to all modules
PROJECT_ROOT_PREFIX = "/home/adrian/as/blogs/nanonets"
UNDEFINED="undefined"
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')

#  raw dataset
RAW_PREFIX = "datasets/SROIE2019-20191212T043930Z-001/SROIE2019/"
raw_dir = os.path.join(PROJECT_ROOT_PREFIX, RAW_PREFIX)

task1train = "0325updated.task1train(626p)"
task2train = "0325updated.task2train(626p)"
task1train_dir = os.path.join(raw_dir, task1train)
task2train_dir = os.path.join(raw_dir, task2train)

# normalized data
NORMALIZED_PREFIX = "invoice-ie-with-gcn/data/normalized/"
normalized_dir = os.path.join(PROJECT_ROOT_PREFIX, NORMALIZED_PREFIX)


### Data Access
# Read raw dataset
# ---------------------------------------------------------------------------------------------------

def get_raw_filepaths(dataset_dir):
    # return full file paths
    images, labels = [], []
    
    for r, d, f in os.walk(dataset_dir):
        for filename in f:
            filepath = os.path.join(dataset_dir, filename)
            if '(' in filename:
                continue
            if filename.endswith(".jpg"):
                images.append(filepath)
            elif filename.endswith(".txt"):
                labels.append(filepath)
    
    return sorted(images), sorted(labels)

def read_raw_word_areas(words_filepath):
    word_areas = []
    with open(words_filepath, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            x0, y0, x1, y1, x2, y2, x3, y3 = row[:8]
            x0, y0, x2, y2 = int(x0), int(y0), int(x2), int(y2)
            content = ','.join(row[8:])
            word_area = WordArea(x0, y0, x2, y2, content, len(word_areas))
            word_areas.append(word_area)
    return word_areas

def read_raw_entities(entities_filepath):
    k2v = {}
    with open(entities_filepath, 'r') as json_file:
        data = json.load(json_file)
        #print(data)
        k2v = {k:v for k, v in data.items()}
        v2k = {v.replace(' ', ''):k for k, v in data.items()}  # TODO: check for collisions
    return k2v, v2k

def load_raw_example(image_filepath, words_filepath, entities_filepath):
    """
    Reads one image and related data files from the original Receipts dataset.
    Reads the entities in json raw format
    """

    # load color (BGR) image
    # BGR is the same as RGB (just inverse byte order), OpenCV adopted it for historical reasons.
    img = cv2.imread(image_filepath)

    # read image words
    word_areas = read_raw_word_areas(words_filepath)

    # read image entities
    _, word2entity = read_raw_entities(entities_filepath)
    
    return img, word_areas, word2entity


# Write dataset to normalized directory
# ---------------------------------------------------------------------------------------------------
def create_filepath(parent_dir, dirname, filename):
    sub_dir = os.path.join(parent_dir,dirname)
    Path(sub_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(sub_dir, filename)
    
def store_normalized_example(target_dir, image_filename, image, word_areas=[], entities=[]):
    """
    save image and related data
    - Word Areas are converted to CSV format
    - Entities are stored in tabular format
    """
    
    # save image
    image_filepath = create_filepath(target_dir, "images", image_filename)
    cv2.imwrite(image_filepath, image)

    # save words
    words_filename = "{}.csv".format(image_filename.split(".")[0])
    words_filepath = create_filepath(target_dir, "words", words_filename)
    with open(words_filepath, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for word_area in word_areas:
            row = list(word_area)[:-1] # drop the index
            csv_writer.writerow(row)   # encode as csv 
        
    # save entities
    entities_filename = "{}.csv".format(image_filename.split(".")[0])
    entities_filepath = create_filepath(target_dir, "entities", entities_filename)
    with open(entities_filepath, 'w') as f:
        for entity in entities:
            f.write(entity + "\n")

### Raw Data Transforms
# Matching the tags to the annotations
# ----------------------------------------------------------------------------
def match_approximate(tags, word_areas, string, tag, tolerance=0.1):
    max_dist = len(string)*tolerance
    for wa in word_areas:
        s = wa.content.replace(' ', '')  
        if editdistance.eval(s, string) <= max_dist:
            tags[wa.idx] = tag
            return True
    return False

def match_truncated(tags, word_areas, string, tag):
    for wa in word_areas:
        s = wa.content.replace(' ', '')
        s = s[:len(string)]   
        if s == string:
            tags[wa.idx] = tag
            return True
    return False

def match_multiple(tags, word_areas, string, tag, tolerance=0.1):
    match = False
    
    for wa in word_areas:
        s = wa.content.replace(' ', '')
        max_dist = len(s)*tolerance
        if len(string) > len(s):
            for pos in range(len(string)-len(s)):
                part = string[pos:pos+len(s)]
                if editdistance.eval(s, part) <= max_dist:
                    tags[wa.idx] = tag
                    match = True
                    string = string.replace(part, '')
        else:
            if editdistance.eval(s, string) <= max_dist:
                tags[wa.idx] = tag
                match = True
                break
            
    return match


def match_last(tags, word_areas, string, tag):
    for i in range(len(word_areas), 0, -1):
        wa = word_areas[i-1]
        s = wa.content.replace(' ', '') 
        if s == string:
            tags[wa.idx] = tag
            return True
    return False

def match_words_to_entities(word_areas, text2tag):
    """
    Match each tag to the closest text item.
    Matches each entitiy to a word and returns the entities in tabular format (normalized entity table)
    """
    # Challenges:
    #  - The company tag value has been spell-checked for typos (match text-items with editdistance<0.1)
    #  - The date tag value is normalized, the text-item is a timestamp (match truncated text-item)
    #  - The address tag value is a join of multiple text-items (match included text-items)
    #  - The total tag value is matched by multiple text-items (match the last text-item)    
    tags = ["undefined"] * len(word_areas)
    
    for string, tag in text2tag.items():
        if "company" in tag:
            match_approximate(tags, word_areas, string, tag)
        elif "date" in tag:
            match_truncated(tags, word_areas, string, tag)
        elif "address" in tag:
            match_multiple(tags, word_areas, string, tag)
        elif "total" in tag:
            match_last(tags, word_areas, string, tag)
       
    return tags


### Run Data Normalization Process
# -------------------------------------------------------------------------
def normalize_dataset(task1train_dir, task2train_dir, target_dir):
    """
    load the images and associated data files from the original "raw" dataset
    and save it in a new format with the entity tags in tabular format
    """

    # find raw files
    image_files, word_files = get_raw_filepaths(task1train_dir)
    _, entity_files = get_raw_filepaths(task2train_dir)

    for i, (i_file, w_file, e_file) in enumerate(zip(image_files, word_files, entity_files)):
        try:
            # read example in raw format
            image, word_areas, word2entity = load_raw_example(i_file, w_file, e_file)

            # Normalize example. 
            # match entity tags to words (approximately) and covert entities to tabular format
            entities = match_words_to_entities(word_areas, word2entity)

            # store normalized example in new location
            image_filename = i_file.split('/')[-1]
            store_normalized_example(target_dir, image_filename, image, word_areas, entities)
        except Exception as e:
            loggin.exception(e)
            logging.info(i_file)
    
    print("Total files: {}".format(i))

# -------------------------------------------------------------------------
def main():

    normalize_dataset(task1train_dir, task2train_dir, normalized_dir)

if __name__ == "__main__":
    main()
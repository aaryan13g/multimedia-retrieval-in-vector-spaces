import os
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import sys
from p3_task1 import create_data_matrix, extract_features_for_new_image


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        np.random.seed(69)
        self.projections = np.random.uniform(low=-0.5, high=0.5, size=(self.hash_size, inp_dimensions))

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector - 0.5, self.projections.T) > 0).astype('int')
        hash = ''.join(bools.astype('str'))
        return hash

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return hash_value


class LSH:
    def __init__(self, n_layers, n_hash_per_layer, inp_dimensions):
        self.n_layers = n_layers
        self.hash_size = n_hash_per_layer
        self.inp_dimensions = inp_dimensions
        self.layers = list()
        self.hash_table = dict()
        for i in range(self.n_layers):
            self.layers.append(HashTable(self.hash_size, self.inp_dimensions))

    def get_hash_key(self, inp_vec):
        hash_key = ""
        for table in self.layers:
            hash_key += table[inp_vec]
        return hash_key

    def make_hash_table(self, hash_key, index):
        self.hash_table.setdefault(hash_key, []).append(index)

    def get_table(self):
        return self.hash_table


def get_input():
    print("Enter space-separated values of 'n_layers', 'n_hash_per_layer', 'folder of images', 'feature model', 'query image', 't':")
    n_layers, n_hash_per_layer, folder, feature_model, query_img, t = input().split(" ")  # Can take feature model as input too here.
    return int(n_layers), int(n_hash_per_layer), folder, feature_model, query_img, int(t)


def create_LSH(inp_dimensions, n_layers, n_hash_per_layer):
    LSH_structure = LSH(n_layers, n_hash_per_layer, inp_dimensions)
    return LSH_structure


def get_hash_key(LSH_structure, data_matrix):
    hash_key_list = list()
    for i in range(len(data_matrix)):
        hash_key = LSH_structure.get_hash_key(data_matrix[i])
        hash_key_list.append(hash_key)
    return hash_key_list


def create_dict(LSH_structure, hash_key_list):
    for i in range(len(hash_key_list)):
        LSH_structure.make_hash_table(hash_key_list[i], i + 1)
    return


def print_LSH_size(Hash_key_table):
    print("Size of LSH structure: ", sys.getsizeof(Hash_key_table), " bytes")


def print_outputs():
    pass


def get_similar_images(labels, query_image, t, Hash_key_table, LSH_structure, n_layers, n_hash_per_layer):
    Hash_key = LSH_structure.get_hash_key(query_image)
    Matches = Hash_key_table[Hash_key]
    # Need to write code for changing n_hash_per_layer if nearest neighbors not found
    matches_list = {}
    for i in range(len(Matches)):
        match_name = labels[Matches[i]]
        matches_list[i] = match_name
    return matches_list


if __name__ == "__main__":
    # n_layers, n_hash_per_layer, folder, feature_model, query_image, t = get_input()
    n_layers, n_hash_per_layer, folder, feature_model, query_image, t = 5, 3, "1000", "elbp", "all/image-cc-1-1.png", 10
    data_matrix, labels = create_data_matrix(folder, feature_model, label_mode="all")
    query_vector = np.array(extract_features_for_new_image("../images/" + query_image, feature_model))
    LSH_structure = create_LSH(len(data_matrix[0]), n_layers, n_hash_per_layer)
    Hash_key_list = get_hash_key(LSH_structure, data_matrix)
    create_dict(LSH_structure, Hash_key_list)
    Hash_key_table = LSH_structure.get_table()
    print_LSH_size(Hash_key_table)
    matches = get_similar_images(labels, query_vector, t, Hash_key_table, LSH_structure, n_layers, n_hash_per_layer)
    print(matches)
    print_outputs()

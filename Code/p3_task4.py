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

    def modify_projections(self):
        self.projections = self.projections[:-1, :]

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector - 0.5, self.projections.T) > 0).astype('int')
        hash = ''.join(bools.astype('str'))
        return hash


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
            hash_key += table.generate_hash(inp_vec)
        return hash_key

    def modify_layers(self):
        self.hash_table.clear()
        for table in self.layers:
            table.modify_projections()
        return

    def make_hash_table(self, hash_key, index):
        self.hash_table.setdefault(hash_key, []).append(index)

    def get_table(self):
        return self.hash_table


def get_input():
    print("Enter space-separated values of 'n_layers', 'n_hash_per_layer', 'folder of images', 'feature model', 'query image', 't':")
    n_layers, n_hash_per_layer, folder, feature_model, query_img, t = input().split(" ") 
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


def get_similar_images_LSH(data_matrix, query_vector, n_layers, n_hash_per_layer, labels, Hash_key_table, LSH_structure, t):
    total_buckets = 0
    matches_list = {}
    while True:
        Hash_key = LSH_structure.get_hash_key(query_vector)
        Matches =[]
        if Hash_key in Hash_key_table:
            Matches = Hash_key_table[Hash_key]
            dist_dict = {}
            for i in range(len(Matches)):
                dist = np.linalg.norm(query_vector - data_matrix[Matches[i] - 1])
                dist_dict[Matches[i]-1] = dist
            sorted_dist_dict = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
            matches_list = {}
            i = 1
            for keys in sorted_dist_dict:
                match_name = labels[keys]
                matches_list[i] = match_name
                i += 1
            print("Ranked Nearest neighbours: ", matches_list)
            print("")
        else:
            print("Nearest Neighbors : None")
        print("Buckets searched at this step:", n_layers * n_hash_per_layer)
        total_buckets += n_layers * n_hash_per_layer
        if len(Matches) < t and n_hash_per_layer > 0:
            n_hash_per_layer -= 1
            print("Not enough nearest numbers found ! Optimizing! Decreasing K to : ", n_hash_per_layer)
            LSH_structure.modify_layers()
            Hash_key_table = make_index_structure(LSH_structure, data_matrix)
        else:
            print("Enough nearest neighbors found! Here are top ", t, ":")
            i = 1
            result_matching_dict = {}
            for match in matches_list:
                print("Rank ", match, ": ", matches_list[match])
                result_matching_dict[match] = matches_list[match]
                if i == t:
                    break
                else:
                    i += 1
            print("\n----------Total number of buckets searched: ", total_buckets, "----------\n")
            print("Total number of unique and overall images considered (it's the same): ", len(matches_list))
            break

    return result_matching_dict, matches_list


def make_index_structure(LSH_structure, data_matrix):
    Hash_key_list = get_hash_key(LSH_structure, data_matrix)
    create_dict(LSH_structure, Hash_key_list)
    Hash_key_table = LSH_structure.get_table()
    print_LSH_size(Hash_key_table)
    return Hash_key_table


def print_FP_and_miss_rates(top_t_matches, all_nearest_matches, query_vector, data_matrix, labels):
    t = len(top_t_matches)
    print("-----------------------------------------------------------------------")
    print("Performing sequential scan to identify true top neighbors...")
    neighbors_distances = {}
    for i in range(len(data_matrix)):
        neighbors_distances[labels[i]] = np.linalg.norm(data_matrix[i] - query_vector)
    neighbors_distances = dict(sorted(neighbors_distances.items(), key=lambda item: item[1]))
    true_neighbors = {}
    j = 1
    for img in neighbors_distances:
        true_neighbors[j] = img
        j += 1
        if j > t:
            break
    print("Top " + str(t) + " true neighbors from sequential scan:")
    for rank in true_neighbors:
        print("Rank ", rank, ": ", true_neighbors[rank])
    top_t_matches_set = set(list(top_t_matches.values()))
    true_neighbors_set = set(list(true_neighbors.values()))
    correct_matches = top_t_matches_set.intersection(true_neighbors_set)
    print("Correct matches: ", len(correct_matches))
    print(correct_matches)
    fp_rate = float(len(set(list(all_nearest_matches.values())) - correct_matches)) / float(t)
    miss_rate = float(t - len(correct_matches)) / float(t)
    print("-----------------------------------------------------------------------")
    print("False Positive Rate: ", fp_rate)
    print("Miss Rate: ", miss_rate)
    print("-----------------------------------------------------------------------")
    

if __name__ == "__main__":
    # n_layers, n_hash_per_layer, folder, feature_model, query_image, t = get_input()
    n_layers, n_hash_per_layer, folder, feature_model, query_image, t = 5, 8, "100", "elbp", "all/image-cc-1-9.png", 20
    data_matrix, labels = create_data_matrix(folder, feature_model, label_mode="all")
    query_vector = np.array(extract_features_for_new_image("../images/" + query_image, feature_model))
    LSH_structure = create_LSH(len(data_matrix[0]), n_layers, n_hash_per_layer)
    Hash_key_table = make_index_structure(LSH_structure, data_matrix)
    top_t_matches, all_nearest_matches = get_similar_images_LSH(data_matrix, query_vector, n_layers, n_hash_per_layer, labels, Hash_key_table, LSH_structure, t)
    print_FP_and_miss_rates(top_t_matches, all_nearest_matches, query_vector, data_matrix, labels)

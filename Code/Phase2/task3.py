import sys
import numpy as np
import pymongo
import os
from task1 import pca, lda, svd, kmeans, condense_matrix, save_latent_semantics
from task2 import print_to_console
import itertools


def fetch_data_matrix(type, model):
    # print("Fetching type data matrix for ", type)
    if "type-" + type + "_" + model + ".csv" not in os.listdir("Data matrices"):
        raise Exception("type-" + type + "_" + model + ".csv not found! Please create first using task1.py!")
    else:
        data_matrix = np.loadtxt("Data matrices/type-" + type + "_" + model + ".csv", delimiter=',')
    return data_matrix


def calc_distance_between_types(type1_matrix, type2_matrix):
    dist_list = []
    for p in range(len(type1_matrix)):
        feature_vector_1 = type1_matrix[p]
        feature_vector_2 = type2_matrix[p]
        dist = np.linalg.norm(feature_vector_1 - feature_vector_2)
        dist_list.append(dist)
    dist_sum = sum(dist_list)
    return dist_sum / len(type1_matrix)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_sys_args():
    return str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    model, k, dim_red = get_sys_args()
    if k > 11:
        k = 11
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures

    if "type-type-similarity-matrix-" + model + ".csv" not in os.listdir("../Outputs/"):
        print("type-type-similarity-matrix-" + model + ".csv not found. Creating it!")
        type_type_similarity_matrix = np.zeros((12, 12))
        all_types = db.distinct("X")
        data_matrix_dict = {}
        for type in all_types:
            data_matrix_dict[type] = fetch_data_matrix(type, model)
        type_combinations = itertools.combinations(all_types, 2)
        i = 0
        j = 0
        for combination in type_combinations:
            while type_type_similarity_matrix[i][j] != 0:
                # print("Value filled: ", i, " ", j)
                if j == 11:
                    i = i + 1
                    j = 0
                else:
                    j = j + 1
            if i == j:
                # print("Equal: ", i, " ", j)
                type_type_similarity_matrix[i][j] = 0.0
                j = j + 1
            dist_val = calc_distance_between_types(data_matrix_dict[combination[0]], data_matrix_dict[combination[1]])
            type_type_similarity_matrix[i][j] = dist_val
            type_type_similarity_matrix[j][i] = dist_val
            # print(combination, " ", i, " ", j, " ", dist_val)
            if j == 11:
                i = i + 1
                j = 0
            else:
                j = j + 1
        # Scaling the similarity matrix and subtracting from 1 to convert distance to similarity.
        type_type_similarity_matrix = normalize_data(type_type_similarity_matrix)
        type_type_similarity_matrix = 1 - type_type_similarity_matrix
        np.savetxt("../Outputs/type-type-similarity-matrix-" + model + ".csv", type_type_similarity_matrix, delimiter=',')
    else:
        print("Existing type-type-similarity-matrix-" + model + ".csv found!")
        type_type_similarity_matrix = np.loadtxt("../Outputs/type-type-similarity-matrix-" + model + ".csv", delimiter=',')
    print("\nType-Type Similarity Matrix:\n", type_type_similarity_matrix)
    WT = None
    LS = None
    if dim_red == "pca":
        LS, S = pca(k, type_type_similarity_matrix)
        WT = type_type_similarity_matrix @ LS
        if k == 11:
            save_latent_semantics(model, "type", dim_red, LS, "LS")
            save_latent_semantics(model, "type", dim_red, S, "S")
            save_latent_semantics(model, "type", dim_red, WT, "WT")

    elif dim_red == "svd":
        WT, VT, S = svd(k, type_type_similarity_matrix)
        LS = VT.transpose()
        if k == 11:
            save_latent_semantics(model, "type", dim_red, LS, "LS")
            save_latent_semantics(model, "type", dim_red, S, "S")
            save_latent_semantics(model, "type", dim_red, WT, "WT")

    elif dim_red == "lda":
        WT = lda(k, type_type_similarity_matrix)
        LS = type_type_similarity_matrix.transpose() @ WT
        if k == 11:
            save_latent_semantics(model, "type", dim_red, LS, "LS")
            save_latent_semantics(model, "type", dim_red, WT, "WT")

    elif dim_red == "kmeans":
        WT = kmeans(k, type_type_similarity_matrix)
        LS = type_type_similarity_matrix.transpose() @ WT
        if k == 11:
            save_latent_semantics(model, "type", dim_red, LS, "LS")
            save_latent_semantics(model, "type", dim_red, WT, "WT")

    print_to_console(WT, model, "type-type")

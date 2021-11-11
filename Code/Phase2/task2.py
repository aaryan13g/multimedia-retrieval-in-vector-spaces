import sys
import numpy as np
import pymongo
import os
from task1 import pca, lda, svd, kmeans, condense_matrix


def save_latent_semantics(model, Y, dim_red, data, data_label):
    # base_folder_path = "../Outputs/"
    # filename = "subject-" + Y + "_" + model + "_" + dim_red + "_" + data_label + ".csv"  # data-label stores whether it is U, V, LS
    # np.savetxt(base_folder_path + filename, data, delimiter=',')  # data stores matrix of U,V,LS..
    pass


def print_to_console(type_wt_pair_matrix, model, Y):
    print("\nType-weight matrix for subject '", Y, "' and '", model, "' features: \n")
    types = ["cc", "con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot", "smooth", "stipple"]
    for i in range(len(type_wt_pair_matrix)):
        print("Type ", types[i], ": ", type_wt_pair_matrix[i])


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    model, Y, k, dim_red = get_sys_args()
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    result = db.find({"Y": Y}, {model: 1, "_id": 0})
    data_matrix = []
    if "subject-" + Y + "_" + model + ".csv" not in os.listdir("Data matrices"):
        for document in result:
            data_matrix.append(document[model])
        data_matrix = np.array(data_matrix)
        np.savetxt("Data matrices/subject-" + Y + "_" + model + ".csv", data_matrix, delimiter=',')
    else:
        data_matrix = np.loadtxt("Data matrices/subject-" + Y + "_" + model + ".csv", delimiter=',')
    WT = None
    LS = None
    if dim_red == "pca":
        LS, S = pca(k, data_matrix)
        WT = data_matrix @ LS
        print(WT)
        WT = condense_matrix(WT)
        if k == 50:
            save_latent_semantics(model, Y, dim_red, LS, "LS")
            save_latent_semantics(model, Y, dim_red, S, "S")
            save_latent_semantics(model, Y, dim_red, WT, "WT")

    elif dim_red == "svd":
        WT, VT, S = svd(k, data_matrix)
        WT = condense_matrix(WT)
        LS = VT.transpose()
        if k == 50:
            save_latent_semantics(model, Y, dim_red, LS, "LS")
            save_latent_semantics(model, Y, dim_red, S, "S")
            save_latent_semantics(model, Y, dim_red, WT, "WT")

    elif dim_red == "lda":
        WT = lda(k, data_matrix)
        LS = data_matrix.transpose() @ WT
        WT = condense_matrix(WT)
        if k == 50:
            save_latent_semantics(model, Y, dim_red, LS, "LS")
            save_latent_semantics(model, Y, dim_red, WT, "WT")

    elif dim_red == "kmeans":
        WT = kmeans(k, data_matrix)
        LS = data_matrix.transpose() @ WT
        WT = condense_matrix(WT)
        if k == 50:
            save_latent_semantics(model, Y, dim_red, LS, "LS")
            save_latent_semantics(model, Y, dim_red, WT, "WT")

    print_to_console(WT, model, Y)

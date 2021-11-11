import sys
import numpy as np
import pymongo
import os
from task1 import pca, lda, svd, kmeans, condense_matrix, print_to_console
from task2 import save_latent_semantics
import itertools


def fetch_data_matrix(subject, model):
    # print("Fetching subject data matrix for ", subject)
    if "subject-" + subject + "_" + model + ".csv" not in os.listdir("Data matrices"):
        raise Exception("subject-" + subject + "_" + model + ".csv not found! Please create first using task1.py!")
    else:
        data_matrix = np.loadtxt("Data matrices/subject-" + subject + "_" + model + ".csv", delimiter=',')
    return data_matrix


def calc_distance_between_subjects(subject1_matrix, subject2_matrix):
    dist_list = []
    for p in range(len(subject1_matrix)):
        feature_vector_1 = subject1_matrix[p]
        feature_vector_2 = subject2_matrix[p]
        dist = np.linalg.norm(feature_vector_1 - feature_vector_2)
        dist_list.append(dist)
    dist_sum = sum(dist_list)
    return dist_sum / len(subject1_matrix)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_sys_args():
    return str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    model, k, dim_red = get_sys_args()
    if k > 39:
        k = 39
    client = pymongo.MongoClient(
        "mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures

    if "subject-subject-similarity-matrix-" + model + ".csv" not in os.listdir("../Outputs/"):
        print("subject-subject-similarity-matrix-" + model + ".csv not found. Creating it!")
        subject_subject_similarity_matrix = np.zeros((40, 40))
        all_subjects = db.distinct("Y")
        all_subjects.sort(key=int)
        data_matrix_dict = {}
        for subject in all_subjects:
            data_matrix_dict[subject] = fetch_data_matrix(subject, model)
        subject_combinations = itertools.combinations(all_subjects, 2)
        i = 0
        j = 0
        for combination in subject_combinations:
            while subject_subject_similarity_matrix[i][j] != 0:
                # print("Value filled: ", i, " ", j)
                if j == 39:
                    i = i + 1
                    j = 0
                else:
                    j = j + 1
            if i == j:
                # print("Equal: ", i, " ", j)
                subject_subject_similarity_matrix[i][j] = 0.0
                j = j + 1
            # print(combination, " ", i, " ", j)
            dist_val = calc_distance_between_subjects(data_matrix_dict[combination[0]],
                                                      data_matrix_dict[combination[1]])
            subject_subject_similarity_matrix[i][j] = dist_val
            subject_subject_similarity_matrix[j][i] = dist_val
            # print(combination, " ", dist_val)
            if j == 39:
                i = i + 1
                j = 0
            else:
                j = j + 1
        subject_subject_similarity_matrix = normalize_data(subject_subject_similarity_matrix)
        subject_subject_similarity_matrix = 1 - subject_subject_similarity_matrix
        np.savetxt("../Outputs/subject-subject-similarity-matrix-" + model + ".csv", subject_subject_similarity_matrix,
                   delimiter=',')
    else:
        print("Existing subject-subject-similarity-matrix-" + model + ".csv found!")
        subject_subject_similarity_matrix = np.loadtxt("../Outputs/subject-subject-similarity-matrix-" + model + ".csv",
                                                       delimiter=',')
    print("\nSubject-Subject Similarity Matrix:\n", subject_subject_similarity_matrix)
    WT = None
    LS = None
    if dim_red == "pca":
        LS, S = pca(k, subject_subject_similarity_matrix)
        WT = subject_subject_similarity_matrix @ LS
        if k == 39:
            save_latent_semantics(model, "subject", dim_red, LS, "LS")
            save_latent_semantics(model, "subject", dim_red, S, "S")
            save_latent_semantics(model, "subject", dim_red, WT, "WT")

    elif dim_red == "svd":
        WT, VT, S = svd(k, subject_subject_similarity_matrix)
        LS = VT.transpose()
        if k == 39:
            save_latent_semantics(model, "subject", dim_red, LS, "LS")
            save_latent_semantics(model, "subject", dim_red, S, "S")
            save_latent_semantics(model, "subject", dim_red, WT, "WT")

    elif dim_red == "lda":
        WT = lda(k, subject_subject_similarity_matrix)
        LS = subject_subject_similarity_matrix.transpose() @ WT
        if k == 39:
            save_latent_semantics(model, "subject", dim_red, LS, "LS")
            save_latent_semantics(model, "subject", dim_red, WT, "WT")

    elif dim_red == "kmeans":
        WT = kmeans(k, subject_subject_similarity_matrix)
        LS = subject_subject_similarity_matrix.transpose() @ WT
        if k == 39:
            save_latent_semantics(model, "subject", dim_red, LS, "LS")
            save_latent_semantics(model, "subject", dim_red, WT, "WT")

    print_to_console(WT, model, "subject-subject")

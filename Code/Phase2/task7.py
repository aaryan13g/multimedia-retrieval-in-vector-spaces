import sys
import os
import numpy as np
import pymongo
from matplotlib import pyplot as plt
from PIL import Image
from math import ceil
import pandas as pd
from Phase1.task1 import color_moments_model, extract_lbp, histogram_of_oriented_gradients_model


def normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def extract_features_for_new_image(image_path):
    image = Image.open(image_path)
    image_data = np.asarray(image) / 255  # normalize the image array
    cm = color_moments_model(image_data)
    cm = normalized(cm.flatten(order="C").tolist()).tolist()
    elbp = normalized(extract_lbp(image_data)).tolist()
    hog, _ = histogram_of_oriented_gradients_model(image_data)
    hog = normalized(hog.tolist()).tolist()
    new_document = {"img_name": image_path, "cm": cm, "elbp": elbp, "hog": hog}
    return new_document


def calculate_distance_with_all_images(query_img_matrix, wt_matrix):
    dist_dict = {}
    for i in range(len(wt_matrix)):
        data_vector = wt_matrix[i]
        dist_dict[i + 1] = np.abs(np.linalg.norm(data_vector - query_img_matrix))
    dist_dict = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
    key_min = min(dist_dict.keys(), key=(lambda k: dist_dict[k]))
    return key_min, dist_dict


def distance_to_similarity(dist_dict):
    similarity_dict = {}
    for subject in dist_dict:
        similarity_dict[subject] = '{0:.8f}'.format((1 / dist_dict[subject]) * 100)
    return similarity_dict


def display_type(dist_dict, similarity_dict, subject_of_query_image):
    final_dict = {key: (dist_dict[key], similarity_dict[key]) for key in similarity_dict}
    distance_similarity = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Distance', 'Similarity'])
    print()
    print(distance_similarity)
    print("\nSubject Label Associated: ", subject_of_query_image, "\n")


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2])


if __name__ == "__main__":

    np.set_printoptions(suppress=True)
    query_img_name, ls_file = get_sys_args()

    client = pymongo.MongoClient(
        "mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    if '/' in query_img_name:
        temp = query_img_name.split('/')[-1]
    else:
        temp = query_img_name
    document = db.find_one({"img_name": temp})
    if document is None:
        document = extract_features_for_new_image(query_img_name)

    base_ls_path = "../Outputs/"
    temp = ls_file.split('_')[-1]
    model = ls_file.split('_')[-2]
    if temp in ("svd", "pca"):
        sigma_file_path = base_ls_path + ls_file + "_S.csv"
        ls_file_path = base_ls_path + ls_file + "_LS.csv"
        wt_file_path = base_ls_path + ls_file + "_WT.csv"

        sigma = np.loadtxt(sigma_file_path, delimiter=',')
        ls = np.loadtxt(ls_file_path, delimiter=',')
        ls_mat = ls @ sigma
        wt = np.loadtxt(wt_file_path, delimiter=',')
        wt_mat = wt @ sigma
    else:
        ls_file_path = base_ls_path + ls_file + "_LS.csv"
        ls_mat = np.loadtxt(ls_file_path, delimiter=',')  # Mxk
        wt_file_path = base_ls_path + ls_file + "_WT.csv"
        wt_mat = np.loadtxt(wt_file_path, delimiter=',')

    query_img_matrix = np.array(document[model])
    query_img_matrix = np.reshape(query_img_matrix, (1, len(query_img_matrix)))
    query_img_matrix_in_latent_features = query_img_matrix @ ls_mat

    subject_of_query_image, dist_dict = calculate_distance_with_all_images(query_img_matrix_in_latent_features, wt_mat)
    similarity_dict = distance_to_similarity(dist_dict)
    display_type(dist_dict, similarity_dict, subject_of_query_image)
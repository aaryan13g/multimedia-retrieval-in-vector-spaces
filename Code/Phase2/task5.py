import sys
import os
import numpy as np
import pymongo
from matplotlib import pyplot as plt
from PIL import Image
from math import ceil
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


def calculate_distance_with_all_images(query_img_matrix, full_data_matrix):
    dist_dict = {}
    for i, img in enumerate(os.listdir("../all/")):
        data_vector = full_data_matrix[i]
        dist_dict[img] = np.abs(np.linalg.norm(data_vector - query_img_matrix))
    dist_dict = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
    return dist_dict


def normalize_dist_dict(dist_dict):
    values = dist_dict.values()
    min_ = min(values)
    max_ = max(values)
    normalized_d = {key: ((v - min_) / (max_ - min_)) for (key, v) in dist_dict.items()}
    return normalized_d


def visualize_nearest_images(base_image, nearest_images):
    fig = plt.figure(figsize=(10, 10))
    rows = ceil((len(nearest_images) + 1) / 2)
    cols = 2
    im = Image.open(base_image)
    fig.add_subplot(rows, cols, 1)
    plt.imshow(im, cmap="gray")
    plt.axis("off")
    plt.title("Query Image: " + base_image)
    i = 2
    for image in nearest_images:
        im = Image.open("../all/" + image)
        fig.add_subplot(rows, cols, i)
        i = i + 1
        plt.imshow(im, cmap="gray")
        plt.axis("off")
        plt.title("Nearest image " + str(i-2) + ": " + image)
        plt.text(64, 32, "Match score: " + '{0:.8f}'.format((1.0 - nearest_images[image]) * 100))

    plt.suptitle("Similar Images for " + base_image)
    plt.show()


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    img_name, ls_file, n = get_sys_args()
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    if '/' in img_name:
        temp = img_name.split('/')[-1]
    else:
        temp = img_name
    document = db.find_one({"img_name": temp})
    if document is None:
        document = extract_features_for_new_image(img_name)
        
    base_ls_path = "../Outputs/"
    temp = ls_file.split('_')[-1]
    model = ls_file.split('_')[-2]
    if temp in ("svd", "pca"):
        sigma_file_path = base_ls_path + ls_file + "_S.csv"
        ls_file_path = base_ls_path + ls_file + "_LS.csv"
        sigma = np.loadtxt(sigma_file_path, delimiter=',')
        ls = np.loadtxt(ls_file_path, delimiter=',')
        ls_mat = ls @ sigma
    else:
        ls_file_path = base_ls_path + ls_file + "_LS.csv"
        ls_mat = np.loadtxt(ls_file_path, delimiter=',')

    full_data_matrix = []
    if "full-data-matrix-" + model + ".csv" not in os.listdir("Data matrices/"):
        print("full-data-matrix-" + model + ".csv not found. Creating it!")
        for img in os.listdir("../all/"):
            print(img)
            res = db.find_one({"img_name": img})
            full_data_matrix.append(res[model])
        full_data_matrix = np.array(full_data_matrix)
        np.savetxt("Data matrices/full-data-matrix-" + model + ".csv", full_data_matrix, delimiter=',')
    else:
        print("Existing full-data-matrix-" + model + ".csv found!")
        full_data_matrix = np.loadtxt("Data matrices/full-data-matrix-" + model + ".csv", delimiter=',')

    print("Full data matrix ", full_data_matrix.shape)
    query_img_matrix = np.array(document[model])
    query_img_matrix = np.reshape(query_img_matrix, (1, len(query_img_matrix)))
    print("Query img matrix ", query_img_matrix.shape)

    data_matrix_in_latent_features = full_data_matrix @ ls_mat
    print("Full data matrix in LS ", data_matrix_in_latent_features.shape)
    img_matrix_in_latent_features = query_img_matrix @ ls_mat
    print("Query img matrix in LS ", img_matrix_in_latent_features.shape)

    dist_dict = calculate_distance_with_all_images(img_matrix_in_latent_features, data_matrix_in_latent_features)
    dist_dict = normalize_dist_dict(dist_dict)
    nearest_images = {}
    for img in dist_dict:
        if n == 0:
            break
        nearest_images[img] = dist_dict[img]
        n = n - 1
    visualize_nearest_images(img_name, nearest_images)

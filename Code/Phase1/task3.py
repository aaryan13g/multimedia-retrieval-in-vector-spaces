import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from PIL import Image


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4])


def get_model_features(feature_descriptors, model):
    feature_dict = {}
    for image in feature_descriptors:
        feature_dict[image] = feature_descriptors[image][model]
    return feature_dict


def calculate_euclidean_distance(base_image, features):
    distances = {}
    for image in features:
        distances[image] = np.linalg.norm(features[base_image] - features[image])
    distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    return distances


def compute_KL_divergence(base_image, image_features):
    kl_distances = {}
    p = image_features[base_image]
    for image in image_features:
        q = image_features[image]
        filt = np.logical_and(p != 0, q != 0)
        kl_distances[image] = np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    return kl_distances


def compute_chi2_distance(base_image, image_features):
    chi2_distances = {}
    image_features[base_image] = image_features[base_image].flatten()
    for image in image_features:
        sm = 0
        image_features[image] = image_features[image].flatten()
        for i in range(len(image_features[base_image])):
            if image_features[base_image][i] + image_features[image][i] == 0:
                sm += 0
            else:
                sm += ((image_features[base_image][i] - image_features[image][i]) ** 2) / (image_features[base_image][i] + image_features[image][i])
        dist = sm / 2
        chi2_distances[image] = dist
    return chi2_distances


def get_nearest_neighbors(distances, k, base_image):
    required_images = {}
    for image in distances:
        if image == base_image:
            continue
        required_images[image] = distances[image]
        k = k - 1
        if k == 0:
            break
    return required_images


def display_most_similar_images(folder, base_image, nearest_images, model=None, contributions=None):
    fig = plt.figure(figsize=(10, 10))
    rows = ceil((len(nearest_images) + 1)/2)
    cols = 2
    im = Image.open(folder + "/" + base_image)
    fig.add_subplot(rows, cols, 1)
    plt.imshow(im, cmap="gray")
    plt.axis("off")
    plt.title("Query Image: " + base_image)
    i = 2
    for image in nearest_images:
        im = Image.open(folder + "/" + image)
        fig.add_subplot(rows, cols, i)
        i = i + 1
        plt.imshow(im, cmap="gray")
        plt.axis("off")
        plt.title("Nearest image " + str(i-2) + ": " + image)
        if contributions is not None:
            plt.text(64, 20, "Match score: " + '{0:.2f}'.format(100.0 - nearest_images[image]), fontsize=10)
            plt.text(64, 30, "Model Contributions:", fontsize=10)
            plt.text(64, 35, "CM: " + str(contributions[image]['CM']) + "%", fontsize=9)
            plt.text(64, 40, "ELBP: " + str(contributions[image]['ELBP']) + "%", fontsize=9)
            plt.text(64, 45, "HOG: " + str(contributions[image]['HOG']) + "%", fontsize=9)
        else:
            plt.text(64, 32, "Match score: " + '{0:.2f}'.format(100.0 - nearest_images[image]))

    if model is None:
        plt.suptitle("Similar Images for Combination of Models")
    else:
        plt.suptitle("Similar Images for " + model + " Model")
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder, image_id, model, k = get_sys_args()
    if model in ("cm", "cm8x8", "CM8x8"):
        model = "CM"
    elif model == "elbp":
        model = "ELBP"
    elif model == "hog":
        model = "HOG"
    f = open('../Outputs/task2_features_' + folder + '.pckl', 'rb')
    feature_descriptors = pickle.load(f)
    f.close()
    all_images_features = get_model_features(feature_descriptors, model)
    distances = calculate_euclidean_distance(image_id, all_images_features)
    nearest_images = get_nearest_neighbors(distances, int(k), image_id)
    display_most_similar_images(folder, image_id, nearest_images, model=model)

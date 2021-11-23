import sys
import pickle
from task3 import get_model_features, calculate_euclidean_distance, get_nearest_neighbors, display_most_similar_images


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])


def compute_weighted_distances(cm, elbp, hog, base_image):
    weighted_distances_dict = {}
    weights_contribution_dict = {}
    for image in cm:
        if image == base_image:
            weighted_distances_dict[image] = 0.0
        else:
            dist_sum = cm[image] + elbp[image] + hog[image]
            w_cm = (dist_sum - cm[image]) / (2 * dist_sum)
            w_elbp = (dist_sum - elbp[image]) / (2 * dist_sum)
            w_hog = (dist_sum - hog[image]) / (2 * dist_sum)
            weighted_dist = (w_cm * cm[image]) + (w_elbp * elbp[image]) + (w_hog * hog[image])
            weighted_distances_dict[image] = weighted_dist
            weights_contribution_dict[image] = {'CM': '{0:.2f}'.format(w_cm * 100), 'ELBP': '{0:.2f}'.format(w_elbp * 100), 'HOG': '{0:.2f}'.format(w_hog * 100)}
    weighted_distances_dict = dict(sorted(weighted_distances_dict.items(), key=lambda item: item[1]))
    return weighted_distances_dict, weights_contribution_dict


if __name__ == "__main__":
    folder, image_id, k = get_sys_args()
    f = open('../Outputs/task2_features_' + folder + '.pckl', 'rb')
    feature_descriptors = pickle.load(f)
    f.close()
    cm_features = get_model_features(feature_descriptors, "CM")
    elbp_features = get_model_features(feature_descriptors, "ELBP")
    hog_features = get_model_features(feature_descriptors, "HOG")
    cm_distances = calculate_euclidean_distance(image_id, cm_features)
    elbp_distances = calculate_euclidean_distance(image_id, elbp_features)
    hog_distances = calculate_euclidean_distance(image_id, hog_features)
    final_distances, contributions = compute_weighted_distances(cm_distances, elbp_distances, hog_distances, image_id)
    nearest_images = get_nearest_neighbors(final_distances, int(k), image_id)
    display_most_similar_images(folder, image_id, nearest_images, contributions=contributions)

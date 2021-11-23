import os
import sys
import pickle
from task1 import fetch_image, color_moments_model, extended_local_binary_patterns_model, histogram_of_oriented_gradients_model


if __name__ == "__main__":
    folder_path = str(sys.argv[1])
    files = os.listdir(folder_path)
    feature_descriptors = {}
    for file in files:
        feature_descriptors[file] = {}
        image_data = fetch_image(folder_path + "/" + file)
        feature_descriptors[file]["CM"] = color_moments_model(image_data)
        feature_descriptors[file]["ELBP"] = extended_local_binary_patterns_model(image_data)
        feature_descriptors[file]["HOG"], _ = histogram_of_oriented_gradients_model(image_data)

    f = open("../Outputs/task2_features_" + folder_path + ".pckl", "wb")
    pickle.dump(feature_descriptors, f)
    f.close()

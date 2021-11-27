import os
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from PIL import Image
from Phase1.task1 import color_moments_model, extract_lbp, histogram_of_oriented_gradients_model
from Phase2.task1 import pca, svd, kmeans, lda
from Phase2.task9 import Node, create_sim_graph, convert_graph_to_nodes, pagerank_one_iter


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.separators = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        separators = {}
        unique_labels = list(set(y))
        for lab in unique_labels:
            print("Finding separator for label: ", lab)
            y_ = [1 if y[l] == lab else -1 for l in range(len(y))]
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        w -= self.lr * (2 * self.lambda_param * w)
                    else:
                        w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y_[idx]))
                        b -= self.lr * y_[idx]
            separators[lab] = (w, b)
            if len(unique_labels) == 2:     # if only binary classification is needed, then no need to train for separators for both classes.
                separators[unique_labels[1]] = (-1 * w, -1 * b)
                break
        self.separators = separators

    def predict(self, X):
        pred_list = [[]] * len(X)
        for separator in self.separators:
            w, b = self.separators[separator]
            approx = np.dot(X, w) - b
            for i in range(len(approx)):
                if approx[i] >= 0:
                    pred_list[i] = list(np.append(pred_list[i], separator))
        final_pred_list = []
        for predicted_labels in pred_list:
            if len(predicted_labels) == 1:
                final_pred_list.append(predicted_labels[0])
                continue
            if len(predicted_labels) == 0:
                predicted_labels = list(self.separators.keys())
            i = 0
            minn = 99999999999999
            nearest_label = ""
            for predicted_label in predicted_labels:
                w, b = self.separators[predicted_label]
                distance = abs(np.dot(X[i], w) - b)
                if distance < minn:
                    nearest_label = predicted_label
                    minn = distance
                i += 1
            final_pred_list.append(nearest_label)
        return final_pred_list


def get_input():
    print("Enter space-separated values of 'train folder', 'feature model', 'k':")
    train_folder, feature_model, k = input().split(" ")
    print("Enter space-separated values of 'test folder', 'classifier':")
    test_folder, classifier = input().split(" ")
    return train_folder, feature_model, k, test_folder, classifier


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def extract_features_for_new_image(image_path, feature_model):
    image = Image.open(image_path)
    image_data = np.asarray(image) / 255  # normalize the image array
    features = None
    if feature_model == 'cm':
        features = color_moments_model(image_data)
        features = normalize_data(features.flatten(order="C").tolist()).tolist()
    elif feature_model == 'elbp':
        features = normalize_data(extract_lbp(image_data)).tolist()
    elif feature_model == 'hog':
        features, _ = histogram_of_oriented_gradients_model(image_data)
        features = normalize_data(features.tolist()).tolist()
    
    return features


def create_data_matrix(folder, feature_model, label_mode):
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    train_path = '../images/' + folder + '/'
    images = [image for image in os.listdir(train_path)]
    data_matrix = []
    labels = []
    if folder + '_' + feature_model + '.csv' not in os.listdir("Data-matrices"):
        print(folder + '_' + feature_model + '.csv not found! Creating and saving it...')
        result = db.find({"img_name": {"$in": images}}, {'img_name': 1, feature_model: 1, label_mode: 1, "_id": 0})
        if len(images) == db.count_documents({"img_name": {"$in": images}}):
            for document in result:
                data_matrix.append(document[feature_model])
                if label_mode == "all":
                    labels.append(document["img_name"])
                else:
                    labels.append(document[label_mode])
        else:
            img_names = [doc['img_name'] for doc in result]
            for image in images:
                if image not in img_names:
                    print('new image ', image)
                    features = extract_features_for_new_image(train_path + image, feature_model)
                    temp = image[:-4].split('-')
                    if label_mode == 'X':
                        label = temp[1]
                    elif label_mode == 'Y':
                        label = temp[2]
                    elif label_mode == 'Z':
                        label = temp[3]
                    else:
                        label = image
                    document = {'img_name': image, feature_model: features, label_mode: label}
                else:
                    document = db.find_one({"img_name": image}, {'img_name': 1, feature_model:1, label_mode: 1, "_id": 0})
                data_matrix.append((document[feature_model]))
                if label_mode == "all":
                    labels.append(document["img_name"])
                else:
                    labels.append(document[label_mode])
        data_matrix = np.array(data_matrix)
        np.savetxt("Data-matrices/" + folder + '_' + feature_model + '.csv', data_matrix, delimiter=',')
    else:
        print(folder + '_' + feature_model + '.csv found!')
        result = db.find({"img_name": {"$in": images}}, {"img_name": 1, label_mode: 1, "_id": 0})
        data_matrix = np.loadtxt("Data-matrices/" + folder + '_' + feature_model + '.csv', delimiter=',')
        if label_mode == "all":
            labels = [document['img_name'] for document in result]
        else:
            labels = [document[label_mode] for document in result]
    print("Shape of data matrix: ", data_matrix.shape)
    print("No. of labels: ", len(labels))
    return data_matrix, labels


def apply_dim_red(data_matrix, k, dim_red='pca'):
    LS, WT = None, None
    if dim_red == "pca":
        LS, WT, S = pca(k, data_matrix)
    elif dim_red == "svd":
        LS, WT, S = svd(k, data_matrix)
    elif dim_red == "lda":
        LS, WT = lda(k, data_matrix)
    elif dim_red == "kmeans":
        LS, WT = kmeans(k, data_matrix)

    print("Latent Semantics shape: ", LS.shape)
    print("Transformed data matrix shape: ", WT.shape)
    return LS, WT


def dtree():
    model_dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=6, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')
    return model_dtree


def svm():
    svm_model = SVC()
    return svm_model


def calc_distance_between_matrices(matrix1, matrix2):
    dist_list = []
    for p in range(len(matrix1)):
        feature_vector_1 = matrix1[p]
        feature_vector_2 = matrix2[p]
        dist = np.linalg.norm(feature_vector_1 - feature_vector_2)
        dist_list.append(dist)
    dist_sum = sum(dist_list)
    return dist_sum / len(matrix1)


def create_similarity_matrix(train_matrix, train_labels):
    sim_dict = {}
    for i in range(len(train_labels)):
        if train_labels[i] not in sim_dict:
            sim_dict[train_labels[i]] = []
        sim_dict[train_labels[i]] += train_matrix[i].tolist()
    for label in sim_dict:
        sim_dict[label] = np.array(sim_dict[label])
    combinations = list(itertools.combinations(sim_dict.keys(), 2))
    sim_mat_size = len(sim_dict.keys())
    similarity_matrix = np.zeros((sim_mat_size, sim_mat_size))
    i, j = 0, 0
    for combination in combinations:
        while similarity_matrix[i][j] != 0:
            # print("Value filled: ", i, " ", j)
            if j == sim_mat_size - 1:
                i = i + 1
                j = 0
            else:
                j = j + 1
        if i == j:
            # print("Equal: ", i, " ", j)
            similarity_matrix[i][j] = 0.0
            j = j + 1
        dist_val = calc_distance_between_matrices(sim_dict[combination[0]], sim_dict[combination[1]])
        similarity_matrix[i][j] = dist_val
        similarity_matrix[j][i] = dist_val
        # print(combination, " ", i, " ", j, " ", dist_val)
        if j == sim_mat_size - 1:
            i = i + 1
            j = 0
        else:
            j = j + 1
    similarity_matrix = normalize_data(similarity_matrix)
    similarity_matrix = 1 - similarity_matrix
    return similarity_matrix, sim_dict


def find_min_distance_with_data_matrix(individual_data_matrix, image_vector):
    dists = []
    for image in individual_data_matrix:
        dists.append(np.linalg.norm(image - image_vector))
    return sum(dists)/len(dists)


def ppr_classifier(similarity_matrix, label_list):
    similarity_graph = create_sim_graph(similarity_matrix, n=3)
    nodes = convert_graph_to_nodes(similarity_graph, len(similarity_matrix))
    query_subjects = ['13']
    i = 0
    while i < 100:
        converge = pagerank_one_iter(nodes, 0.15, query_subjects)
        if converge:
            break
        i = i + 1
    print("PageRank converged in ", i, " iterations.")
    for node in nodes:
        node.print_node()
    match_dict = {}
    maxx = -1
    max_node = None
    for node in nodes:
        match_dict[node.id] = node.pagerank
        if node.pagerank > maxx:
            maxx = node.pagerank
            max_node = node.id
    match_dict = dict(sorted(match_dict.items(), key=lambda item: item[1]))
    print(match_dict)
    return str(max_node)


def train_classifier(train_matrix, labels, classifier):
    model = None
    if classifier == "dtree":
        model = dtree()
    elif classifier == "svm":
        model = SVM()
    model.fit(train_matrix, labels)
    return model


def predict(model, test_matrix):
    labels = model.predict(test_matrix)
    return labels


def compute_and_print_outputs(true_labels, pred_labels):
    print("True Labels | Predicted Labels")
    for true, pred in zip(true_labels, pred_labels):
        print(true + '\t | \t' + pred)
    print("The accuracy is :\n")
    print(accuracy_score(true_labels, pred_labels))

    cnf_matrix = confusion_matrix(true_labels, pred_labels)  # sklearn function

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    FPR = FP / (FP + TN)  # This is the false positive rate
    FNR = FN / (TP + FN)  # This is the misses

    print(cnf_matrix)


def shuffle(data_matrix, labels):
    labels = np.array(labels)
    indices = np.arange(data_matrix.shape[0])
    np.random.shuffle(indices)
    data_matrix = data_matrix[indices]
    labels = labels[indices].tolist()
    return data_matrix, labels


if __name__ == "__main__":
    # train_folder, feature_model, k, test_folder, classifier = get_input()
    train_folder, feature_model, k, test_folder, classifier = "all", "elbp", "*", "500", "svm"
    data_matrix, labels = create_data_matrix(train_folder, feature_model, label_mode='X')
    data_matrix, labels = shuffle(data_matrix, labels)
    if k != 'all' and k != '*':
        if train_folder + '_' + feature_model + '_' + k + '_LS.csv' in os.listdir('Latent-Semantics') and train_folder + '_' + feature_model + '_' + k + '_WT.csv' in os.listdir('Latent-Semantics'):
            print("Existing latent semantics and train matrix found!")
            latent_semantics = np.loadtxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_LS.csv', delimiter=',')
            train_matrix = np.loadtxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_WT.csv', delimiter=',')
        else:
            print("Existing latent semantics and train matrix not found! Creating and saving them...")
            latent_semantics, train_matrix = apply_dim_red(data_matrix, k)
            np.savetxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_LS.csv', latent_semantics, delimiter=',')
            np.savetxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_WT.csv', train_matrix, delimiter=',')
        print("Latent Semantics File: ", train_folder + '_' + feature_model + '_' + k + '_LS.csv')
    else:
        train_matrix = data_matrix
        latent_semantics = None
    print("Training model now...")
    if classifier == 'ppr':
        similarity_matrix, individual_train_dict = create_similarity_matrix(train_matrix, labels)
        print("Similarity matrix:\n", similarity_matrix)
        test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model, label_mode='X')
        if k != 'all' and k != '*':
            test_matrix = test_data_matrix @ latent_semantics
        else:
            test_matrix = test_data_matrix
        pred_labels = []
        for img in test_matrix[:]:
            distance_list = []
            for label in individual_train_dict:
                distance_list.append(find_min_distance_with_data_matrix(individual_train_dict[label], img))
            similarity_list = normalize_data(distance_list)
            similarity_list = 1 - similarity_list
            updated_similarity_matrix = similarity_matrix.copy()
            updated_similarity_matrix = np.insert(updated_similarity_matrix, len(similarity_list), similarity_list, axis=1)
            similarity_list = np.append(similarity_list, 1)
            updated_similarity_matrix = np.insert(updated_similarity_matrix, len(similarity_list) - 1, similarity_list, axis=0)
            print("Updated Similarity Matrix:\n", updated_similarity_matrix)
            predicted = ppr_classifier(updated_similarity_matrix, list(individual_train_dict.keys()))
            pred_labels.append(predicted)
    else:
        model = train_classifier(train_matrix, labels, classifier)
        print(classifier, " model training completed!")
        test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model, label_mode='X')
        if k != 'all' and k != '*':
            test_matrix = test_data_matrix @ latent_semantics
        else:
            test_matrix = test_data_matrix
        pred_labels = predict(model, test_matrix)
    compute_and_print_outputs(true_labels, pred_labels)
    print("Task Completed!")

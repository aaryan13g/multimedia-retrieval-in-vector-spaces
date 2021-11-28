import os
import pymongo
import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from Phase1.task1 import color_moments_model, extract_lbp, histogram_of_oriented_gradients_model
from Phase2.task1 import pca, svd, kmeans, lda
from Phase2.task9 import Node, create_sim_graph, convert_graph_to_nodes, pagerank_one_iter


class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value


class DTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.unique_labels = None

        
    @staticmethod
    def _entropy(s):
        '''
        Helper function, calculates entropy from an array of integer values.
        
        :param s: list
        :return: float, entropy value
        '''
        # Convert to integers to avoid runtime errors
        counts = np.bincount(np.array(s, dtype=np.int64))
        # Probabilities of each class label
        percentages = counts / len(s)

        # Caclulate entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):
        '''
        Helper function, calculates information gain from a parent and two child nodes.
        
        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        '''
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        # One-liner which implements the previously discussed formula
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
        '''
        Helper function, calculates the best split for given features and target
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        # For every dataset feature
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    # Calculate the information gain and save the split parameters
                    # if the current split if better then the previous best
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape
        
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gain=best['gain']
                )
        # Leaf node - value is the most common target value 
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fitting(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        
    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        self.unique_labels = list(set(y))
        y = np.array([self.unique_labels.index(lab) for lab in y])       
        # Call a recursive function to build the tree
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # Leaf node
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        
        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        '''
        Function used to classify new instances.
        
        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        results = [int(self._predict(x, self.root)) for x in X]
        labels = [self.unique_labels[results[i]] for i in range(len(results))]
        return labels


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


# def dtree():
#     model_dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=6, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')
#     return model_dtree


def calc_distance_between_matrices(matrix1, matrix2):
    # dist_list = []
    sum1 = np.sum(matrix1, axis=0) / len(matrix1)
    sum2 = np.sum(matrix2, axis=0) / len(matrix2)
    dist = np.linalg.norm(sum1 - sum2)
    return dist
    # for v1 in matrix1:
    #     for v2 in matrix2:
    #         dist = np.linalg.norm(v1 - v2)
    #         dist_list.append(dist)
    # for p in range(len(matrix1)):
    #     feature_vector_1 = matrix1[p]
    #     feature_vector_2 = matrix2[p]
    #     dist = np.linalg.norm(feature_vector_1 - feature_vector_2)
    #     dist_list.append(dist)
    # dist_sum = sum(dist_list)
    # return dist_sum / len(matrix1)


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
        print("Calculating similarity between", combination[0], "and", combination[1])
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
    return min(dists)


def ppr_classifier(similarity_matrix, label_list):
    similarity_graph = create_sim_graph(similarity_matrix, label_list, n=3)
    nodes = convert_graph_to_nodes(similarity_graph, len(similarity_matrix), label_list)
    query_subjects = ['query']
    i = 0
    while i < 100:
        converge = pagerank_one_iter(nodes, 0.15, query_subjects)
        if converge:
            break
        i = i + 1
    print("PageRank converged in ", i, " iterations.")
    # for node in nodes:
    #     nodes[node].print_node()
    match_dict = {}
    maxx = -1
    max_node = None
    for node in nodes:
        match_dict[nodes[node].id] = nodes[node].pagerank
        if nodes[node].pagerank > maxx and nodes[node].id != 'query':
            maxx = nodes[node].pagerank
            max_node = nodes[node].id
    match_dict = dict(sorted(match_dict.items(), key=lambda item: item[1]))
    print(match_dict)
    return max_node


def train_classifier(train_matrix, labels, classifier):
    model = None
    if classifier == "dtree":
        model = DTree()
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
    np.random.seed(69)
    np.random.shuffle(indices)
    data_matrix = data_matrix[indices]
    labels = labels[indices].tolist()
    return data_matrix, labels


if __name__ == "__main__":
    # train_folder, feature_model, k, test_folder, classifier = get_input()
    train_folder, feature_model, k, test_folder, classifier = "500", "elbp", "50", "100", "dtree"
    data_matrix, labels = create_data_matrix(train_folder, feature_model, label_mode='X')
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
    train_matrix, labels = shuffle(train_matrix, labels)
    print("Training model now...")
    if classifier == 'ppr':
        similarity_matrix, individual_train_dict = create_similarity_matrix(train_matrix, labels)
        test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model, label_mode='X')
        if k != 'all' and k != '*':
            test_matrix = test_data_matrix @ latent_semantics
        else:
            test_matrix = test_data_matrix
        pred_labels = []
        for img in test_matrix:
            distance_list = []
            for label in individual_train_dict:
                distance_list.append(find_min_distance_with_data_matrix(individual_train_dict[label], img))
            similarity_list = normalize_data(distance_list)
            similarity_list = 1 - similarity_list
            updated_similarity_matrix = similarity_matrix.copy()
            updated_similarity_matrix = np.insert(updated_similarity_matrix, len(similarity_list), similarity_list, axis=1)
            similarity_list = np.append(similarity_list, 1)
            updated_similarity_matrix = np.insert(updated_similarity_matrix, len(similarity_list) - 1, similarity_list, axis=0)
            predicted = ppr_classifier(updated_similarity_matrix, list(individual_train_dict.keys()) + ['query'])
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
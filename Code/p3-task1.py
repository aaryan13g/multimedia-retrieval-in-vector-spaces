import os
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
from sklearn.svm import LinearSVC

def get_input():
    print("Enter space-separated values of 'train folder', 'feature model', 'k':")
    train_folder, feature_model, k = input().split(" ")
    print("Enter space-separated values of 'test folder', 'classifer':")
    test_folder, classifier = input().split(" ")
    return train_folder, feature_model, k, test_folder, classifier


def create_data_matrix(folder, feature_model, label_mode):
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    train_path = '../images/' + folder + '/'
    images = [image for image in os.listdir(train_path)]
    data_matrix = []
    labels = []
    if folder + '_' + feature_model + '.csv' not in os.listdir("Data-matrices"):
        result = db.find({"img_name": {"$in": images}}, {feature_model: 1, label_mode: 1, "_id": 0})
        for document in result:
            data_matrix.append(document[feature_model])
            labels.append(document[label_mode])
        data_matrix = np.array(data_matrix)
        np.savetxt("Data-matrices/" + folder + '_' + feature_model + '.csv', data_matrix, delimiter=',')
    else:
        result = db.find({"img_name": {"$in": images}}, {label_mode: 1, "_id": 0})
        data_matrix = np.loadtxt("Data-matrices/" + folder + '_' + feature_model + '.csv', delimiter=',')
        labels = [document[label_mode] for document in result]
    return data_matrix, labels


def train_classifier(train_matrix, labels, classifier):
    if classifier == "Decison_Trees":
        model = dtree()
    elif classifier == "svm":
        model= svm()
    model.fit(train_matrix, labels)
    return model


def predict(model, test_matrix):
    labels = model.predict(test_matrix)
    return labels


def compute_and_print_outputs(true_labels, pred_labels):
    print("The true labels are :\n")
    print(true_labels)
    print("The predicted labels are :\n")
    print(pred_labels)
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

    # cnf_matrix.print_stats()


def dtree():
    model_dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                    max_features=None, max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=6,
                    min_weight_fraction_leaf=0.0, presort=False,
                    random_state=None, splitter='best')
    # model_dtree.fit(x_train_tfidf,y_train)
    return model_dtree

def svm():
    svm_model = LinearSVC()
    return svm_model

if __name__ == "__main__":
    train_folder, feature_model, k, test_folder, classifier = get_input()
    data_matrix, labels = create_data_matrix(train_folder, feature_model, label_mode='X')
    latent_semantics = apply_dim_red(data_matrix, k)
    train_matrix = transform_into_latent_space(data_matrix, latent_semantics)
    model = train_classifier(train_matrix, labels, classifier)
    print(classifier, " model training completed!")
    test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model, label_mode='X')
    test_matrix = transform_into_latent_space(test_data_matrix, latent_semantics)
    pred_labels = predict(model, test_matrix)
    compute_and_print_outputs(true_labels, pred_labels)
    print("Task Completed!")

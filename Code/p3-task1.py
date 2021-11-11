import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix


def get_input():
    print("Enter space-separated values of 'train folder', 'feature model', 'k':")
    train_folder, feature_model, k = input().split(" ")
    print("Enter space-separated values of 'test folder', 'classifer':")
    test_folder, classifier = input().split(" ")
    return train_folder, feature_model, k, test_folder, classifier


if __name__ == "__main__":
    train_folder, feature_model, k, test_folder, classifier = get_input()
    data_matrix, labels = create_data_matrix(train_folder, feature_model)
    latent_semantics = apply_dim_red(data_matrix, k)
    train_matrix = transform_into_latent_space(data_matrix, latent_semantics)
    model = train_classifier(train_matrix, labels, classifier)
    print(classifier, " model training completed!")
    test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model)
    test_matrix = transform_into_latent_space(test_data_matrix, latent_semantics)
    pred_labels = predict(model, test_matrix)
    compute_and_print_outputs(true_labels, pred_labels)
    print("Task Completed!")

def train_classifier(train_matrix, labels, classifier):
    classifier=classifier_function_name()
    model=classifier.fit(train_matrix,labels)
    return model

def predict(model, test_matrix):
    labels=model.predict(test_matrix)
    return labels

def compute_and_print_outputs(true_labels, pred_labels):
    print("The true labels are :\n")
    print(true_labels)
    print("The predicted labels are :\n")
    print(pred_labels)
    print("The accuracy is :\n")
    print(accuracy_score(true_labels, pred_labels))
    
    cnf_matrix = confusion_matrix(true_labels,pred_labels)  #sklearn function
        
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    FPR = FP/(FP+TN)       #This is the false positive rate
    FNR = FN/(TP+FN)        #This is the misses
    
    # cnf_matrix.print_stats()

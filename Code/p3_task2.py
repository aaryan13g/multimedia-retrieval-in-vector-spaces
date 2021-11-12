import os
import numpy as np
from p3_task1 import create_data_matrix, get_input, apply_dim_red, train_classifier, predict, compute_and_print_outputs

if __name__ == "__main__":
    train_folder, feature_model, k, test_folder, classifier = get_input()
    data_matrix, labels = create_data_matrix(train_folder, feature_model, label_mode='Y')
    if train_folder + '_' + feature_model + '_' + k + '_LS.csv' in os.listdir('Latent-Semantics') and train_folder + '_' + feature_model + '_' + k + '_WT.csv' in os.listdir('Latent-Semantics'):
        print("Existing latent semantics and train matrix found!")
        latent_semantics = np.loadtxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_LS.csv', delimiter=',')
        train_matrix = np.loadtxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_WT.csv', delimiter=',')
    else:
        print("Existing latent semantics and train matrix not found! Creating and saving them...")
        latent_semantics, train_matrix = apply_dim_red(data_matrix, k)
        np.savetxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_LS.csv', latent_semantics, delimiter=',')
        np.savetxt('Latent-Semantics/' + train_folder + '_' + feature_model + '_' + k + '_WT.csv', train_matrix, delimiter=',')

    print("Training model now...")
    model = train_classifier(train_matrix, labels, classifier)
    print(classifier, " model training completed!")
    test_data_matrix, true_labels = create_data_matrix(test_folder, feature_model, label_mode='Y')
    test_matrix = test_data_matrix @ latent_semantics
    pred_labels = predict(model, test_matrix)
    compute_and_print_outputs(true_labels, pred_labels)
    print("Task Completed!")
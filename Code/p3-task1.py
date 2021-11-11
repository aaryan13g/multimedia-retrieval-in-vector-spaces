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

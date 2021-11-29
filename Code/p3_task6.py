import numpy as np
from p3_task1 import DTree


def create_relevance_space(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    relevance_train_data_matrix = []
    relevance_train_labels = []
    relevance_image_names = []
    for rank in relevant_imgs:
        relevant_nearest_image_name = nearest_neighbors[rank]
        label_idx = labels.index(relevant_nearest_image_name)
        relevant_image_vector = vector_space_matrix[label_idx]
        relevance_train_data_matrix.append(relevant_image_vector)
        relevance_train_labels.append("relevant")
        relevance_image_names.append(relevant_nearest_image_name)
        del(nearest_neighbors[rank])
    for rank in irrelevant_imgs:
        irrelevant_nearest_image_name = nearest_neighbors[rank]
        label_idx = labels.index(irrelevant_nearest_image_name)
        irrelevant_image_vector = vector_space_matrix[label_idx]
        relevance_train_data_matrix.append(irrelevant_image_vector)
        relevance_train_labels.append("irrelevant")
        relevance_image_names.append(irrelevant_nearest_image_name)
        del(nearest_neighbors[rank])
    relevance_test_data_matrix = []
    for rank in nearest_neighbors:
        test_nearest_image_name = nearest_neighbors[rank]
        label_idx = labels.index(test_nearest_image_name)
        test_image_vector = vector_space_matrix[label_idx]
        relevance_test_data_matrix.append(test_image_vector)
        relevance_image_names.append(test_nearest_image_name)
    return np.array(relevance_train_data_matrix), relevance_train_labels, np.array(relevance_test_data_matrix), relevance_image_names


def dtree_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    relevance_train_data_matrix, relevance_train_labels, relevance_test_data_matrix, relevance_image_names = create_relevance_space(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)
    model = DTree()
    model.fit(relevance_train_data_matrix, relevance_train_labels)
    predicted_labels = model.predict(relevance_test_data_matrix)
    final_data_matrix = np.vstack((relevance_train_data_matrix, relevance_test_data_matrix))
    final_data_labels = relevance_train_labels + predicted_labels
    p = 1
    q = len(final_data_matrix)
    ranked_results = {}
    for i in range(len(final_data_matrix)):
        img_name = relevance_image_names[i]
        if final_data_labels[i] == 'relevant':
            ranked_results[p] = img_name
            p += 1
        else:
            ranked_results[q] = img_name
            q -= 1
    ranked_resultss = {}
    for i in range(len(ranked_results)):
        ranked_resultss[i + 1] = ranked_results[i + 1]
    return ranked_resultss
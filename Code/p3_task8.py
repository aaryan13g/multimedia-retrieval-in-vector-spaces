import numpy as np
import matplotlib.pyplot as plt
from p3_task1 import create_data_matrix, apply_dim_red, extract_features_for_new_image
from p3_task4 import create_LSH, get_similar_images_LSH, make_index_structure
from p3_task7 import svm_relevance_feedback
from p3_task6 import dtree_relevance_feedback
from p3_task5 import *

def create_vector_space(input_folder, feature_model, dim_red, k):
    data_matrix, labels = create_data_matrix(input_folder, feature_model, label_mode='all')
    if dim_red == 'none' and k == '*':
        print("No dimensionality reduction requested!")
        return data_matrix, labels, None
    else:
        print("Applying dimensionality reduction:", dim_red)
        latent_semantics, reduced_matrix = apply_dim_red(data_matrix, k, dim_red)
        return reduced_matrix, labels, latent_semantics


def transform_query(query_img, feature_model, latent_semantics):
    img_path = '../images/' + query_img
    feature_vector = np.array(extract_features_for_new_image(img_path, feature_model))
    if latent_semantics is not None:
        feature_vector = feature_vector @ latent_semantics
    return feature_vector


def run_LSH(L, K, vector_space_matrix, labels, query_img_vector, t):
    LSH_structure = create_LSH(len(vector_space_matrix[0]), L, K)
    Hash_key_table = make_index_structure(LSH_structure, vector_space_matrix)
    top_t_matches, _ = get_similar_images_LSH(vector_space_matrix, query_img_vector, L, K, labels, Hash_key_table, LSH_structure, t)
    return top_t_matches

def run_VA(bits, vector_space_matrix, labels, query_img_vector, t):
    query_img_vector=np.array([query_img_vector])
    res, p = create_VA(vector_space_matrix, bits)
    hashed=create_hash(res)
    li=[]
    temp1=[]
    for j in range(len(vector_space_matrix)):
        chunks = [res[j][i:i+bits] for i in range(0, len(res[j]), bits)]
        temp=[]
        for i in range(len(chunks)):
            temp.append(int(chunks[i],2))
        temp1.append(temp)
        li.append(get_bounds(query_img_vector,temp,p,bits))
        chunks = list(map(int, chunks))
    count=0 
    all_nearest_images={}
    d,ans,count,all_nearest_images=va_ssa(query_img_vector,temp1,t,li,count,all_nearest_images)
    similar_images=get_similar_images(ans,labels)
    return similar_images


def SVM_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    return svm_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)


def DT_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    return dtree_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)


if __name__ == "__main__":
    print("Enter the query image:")
    query_img = input()
    query_img = '../images/' + query_img
    print("Enter number of nearest neighbors required:")
    t = int(input())
    print('-------------------------------------------------------------------------------')
    print("Enter the details of vector set you want: ")
    print("Enter the image folder for creating vector space:")
    input_folder = input()
    # input_folder = '../images/' + input_folder
    print("Enter the feature-model required (cm/elbp/hog):")
    feature_model = input()
    print("Enter the dimensionality reduction model (pca/svd/lda/kmeans/none)")
    dim_red = input()
    if dim_red != "none":
        print("Enter the latent features needed (k):")
        k = int(input())
    else:
        k = '*'
    print("Creating vector space...Please wait.")
    vector_space_matrix, labels, latent_semantics = create_vector_space(input_folder, feature_model, dim_red, k)
    query_img_vector = transform_query(query_img, feature_model, latent_semantics)
    print('-------------------------------------------------------------------------------')
    print("Enter Index-tool (LSH/VA):")
    index_tool = input()
    print("Enter (space-separated) index tool parameters:")
    if index_tool == "LSH":
        print("#Layers(L) #Hash-per-layer(K)")
        inp = list(map(int, input().split()))
        L, K = inp[0], inp[1]
        nearest_neighbors = run_LSH(L, K, vector_space_matrix, labels, query_img_vector, t)
    else:
        print("#Bits-per-dimension(b)")
        b = int(input())
        nearest_neighbors = run_VA(b, vector_space_matrix, labels, query_img_vector, t)
    print('-------------------------------------------------------------------------------')
    # for index in nearest_neighbors:
    #     print('Rank ', index, ':', nearest_neighbors[index])
    # print('-------------------------------------------------------------------------------')
    print("Enter space-separated index numbers for relevant images:")
    while True:
        relevant_imgs = list(map(int, input().split()))
        if len(relevant_imgs) == 0:
            print("You must enter atleast 1 relevant image!")
        else:
            break
    print("Enter space-separated index numbers for irrelevant images:")
    while True:
        irrelevant_imgs = list(map(int, input().split()))
        if len(irrelevant_imgs) == 0:
            print("You must enter atleast 1 irrelevant image!")
        else:
            break
    print('-------------------------------------------------------------------------------')
    print("Which classifier do you want to use for relevance feedback? (svm/dtree):")
    classifier = input()
    if classifier == "svm":
        ranked_results = SVM_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)
    else:
        ranked_results = DT_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)
    print('-------------------------------------------------------------------------------')
    for index in ranked_results:
        print('Rank ', index, ':', ranked_results[index])
    print('-------------------------------------------------------------------------------')

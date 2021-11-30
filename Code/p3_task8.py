import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image
from p3_task1 import create_data_matrix, apply_dim_red, extract_features_for_new_image
from p3_task4 import create_LSH, get_similar_images_LSH, make_index_structure
from p3_task7 import svm_relevance_feedback
from p3_task6 import dtree_relevance_feedback
from p3_task5 import *

def normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) 

def create_vector_space(input_folder, feature_model, dim_red, k,query_img_vector):
    query_img_vector =np.array([query_img_vector])
    data_matrix, labels = create_data_matrix(input_folder, feature_model, label_mode='all')
    data_matrix = np.concatenate((data_matrix, query_img_vector))
    
    if dim_red == 'none' and k == '*':
        print("No dimensionality reduction requested!")
        return data_matrix, labels, None
    else:
        print("Applying dimensionality reduction:", dim_red)
        latent_semantics, reduced_matrix = apply_dim_red(data_matrix, k, dim_red)
    
    for i in range(len(reduced_matrix)):
        reduced_matrix[i] =normalized(reduced_matrix[i])
    
    reduced_query_matrix = reduced_matrix[-1]
    reduced_data_matrix = reduced_matrix[:-1]
    return reduced_data_matrix, reduced_query_matrix, labels, latent_semantics


def transform_query(query_img, feature_model):
    img_path = '../images/' + query_img
    feature_vector = np.array(extract_features_for_new_image(img_path, feature_model))
    return feature_vector


def run_LSH(L, K, vector_space_matrix, labels, query_img_vector, t):
    LSH_structure = create_LSH(len(vector_space_matrix[0]), L, K)
    Hash_key_table = make_index_structure(LSH_structure, vector_space_matrix)
    top_t_matches, _ = get_similar_images_LSH(vector_space_matrix, query_img_vector, L, K, labels, Hash_key_table, LSH_structure, t)
    return top_t_matches

def run_VA(bits, vector_space_matrix, labels, query_img_vector, t):
    query_img_vector=np.array([query_img_vector])
    res, p = create_VA(vector_space_matrix, bits)
    # hashed=create_hash(res)
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
    d,ans,count,all_nearest_images=va_ssa(query_img_vector,temp1,t,li,count,all_nearest_images,bits,vector_space_matrix,labels)
    similar_images=get_similar_images(ans,labels)
    
    print("-------------------------------------------------------------------------------------------------")   
    ani = {}
    i = 1
    for keys in all_nearest_images:
        ani[i] = all_nearest_images[keys]
        i = i + 1

    for keys in ani:
        print("Rank ", keys, " :", ani[keys])
    print("-------------------------------------------------------------------------------------------------")

    print("\nThe nearest images are : \n")
    for keys in similar_images:
        print("Rank ", keys, " :", similar_images[keys])

    print("-------------------------------------------------------------------------------------------------")
    
    return similar_images


def SVM_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    return svm_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)


def DT_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels):
    return dtree_relevance_feedback(relevant_imgs, irrelevant_imgs, nearest_neighbors, vector_space_matrix, labels)


def display_nearest_neighbors(input_folder, nearest_neighbors, mode):
    fig = plt.figure(figsize=(10, 10))
    rows = ceil((len(nearest_neighbors) + 1) / 4)
    cols = 4
    i = 1
    for rank in nearest_neighbors:
        im = Image.open("../images/" + input_folder + "/" + nearest_neighbors[rank])
        fig.add_subplot(rows, cols, i)
        i = i + 1
        plt.imshow(im, cmap="gray")
        plt.axis("off")
        plt.title("Neighbor " + str(i - 1) + ":\n" + nearest_neighbors[rank])
    if mode == "initial":
        plt.suptitle("Nearest Neighbors Without Relevance Feedback")
    elif mode == "reranked":
        plt.suptitle("Re-Ranked Nearest Neighbors With Relevance Feedback")
    plt.show()


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
    query_img_vec = transform_query(query_img, feature_model)
    vector_space_matrix, query_img_vector, labels, latent_semantics = create_vector_space(input_folder, feature_model, dim_red, k,query_img_vec)
    #query_img_vector = transform_query(query_img, feature_model, latent_semantics)
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
    display_nearest_neighbors(input_folder, nearest_neighbors, mode="initial")
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
    print("\nRe-Ranked results based on relevance feedback:\n")
    for index in ranked_results:
        print('Rank ', index, ':', ranked_results[index])
    print('-------------------------------------------------------------------------------')
    display_nearest_neighbors(input_folder, ranked_results, mode="reranked")

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
    input_folder = '../images/' + input_folder
    print("Enter the feature-model required (cm/elbp/hog):")
    feature_model = input()
    print("Enter the dimensionality reduction model (pca/svd/lda/kmeans/none)")
    dim_red = input()
    if dim_red != "none":
        print("Enter the latent features needed (k):")
        k = int(input())
    else:
        k = '*'
    vector_space_matrix, latent_semantics = create_vector_space(input_folder, feature_model, dim_red, k)
    query_img_vector = transform_query(query_img, feature_model, latent_semantics)
    print('-------------------------------------------------------------------------------')
    print("Enter Index-tool (LSH/VA):")
    index_tool = input()
    print("Enter (space-separated) index tool parameters:")
    if index_tool == "LSH":
        print("#Layers(L) #Hash-per-layer(K)")
        inp = list(map(int, input().split()))
        L, K = inp[0], inp[1]
        nearest_neighbors = run_LSH(L, K, vector_space_matrix, query_img_vector, t)
        # nearest_neighbors = {Rank_index: (neighbor_img1, match_score), ......}
    else:
        print("#Bits-per-dimension")
        b = int(input)
        nearest_neighbors = run_VA(b, vector_space_matrix, query_img_vector, t)
    for index in nearest_neighbors:
        print('Index ', index, ':', nearest_neighbors[index][0], ': Match score = ', nearest_neighbors[index][1])
    print("Enter space-separated index numbers for relevant images:")
    relevant_imgs = list(map(int, input().split()))
    print("Enter space-separated index numbers for irrelevant images:")
    irrelevant_imgs = list(map(int, input().split()))


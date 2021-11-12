import sys
import numpy as np
import pymongo
import os
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def pca(k, data_matrix):
    # eigVal = {}
    # data_matrix = data_matrix.transpose()
    # covariance_matrix = np.cov(data_matrix)
    # values, vectors = np.linalg.eigh(covariance_matrix)
    # i = 1  # create a dictionary with i as key which is tracing eigen value
    # for item in values:
    #     eigVal[i] = item
    #     i += 1
    #
    # eigVal = sorted(eigVal.items(), key=lambda x: x[1], reverse=True)  # Sorting for dim reduction to choose top k
    # final_k_eigval_list = []  # stores top k key,eigenvalue pairs
    # for index, tuple in enumerate(eigVal):  # Choosing top k eigen values and storing in a list
    #     if (index == k):
    #         break
    #     final_k_eigval_list.append(tuple)
    #
    # # Extracting k eig vectors
    # final_eigvec_list = []
    # final_k_eigvec_list = []
    # i = 0
    # vec_transpose = vectors.transpose()
    # for tuple in eigVal:
    #     final_eigvec_list.append(vec_transpose[tuple[0] - 1])
    #     i = i + 1
    # final_eigvec_list = np.array(final_eigvec_list).transpose()
    # for i in range(len(final_eigvec_list)):
    #     final_k_eigvec_list.append(final_eigvec_list[i][0:k])
    #
    # return np.array(final_k_eigvec_list), np.diag(np.array(final_k_eigval_list))
    data_matrix_t = data_matrix.transpose()
    pcaa = PCA(n_components=int(k), copy=False)
    latent_semantics = pcaa.fit_transform(data_matrix_t)
    S = np.diag(pcaa.explained_variance_)
    WT = data_matrix @ latent_semantics
    return np.array(latent_semantics), WT, S


def lda(k, data_matrix):
    model = LatentDirichletAllocation(n_components=int(k), learning_method='batch', n_jobs=-1)
    lda_transformed = model.fit_transform(data_matrix)
    data_matrix_t = data_matrix.transpose()
    LS = data_matrix_t @ lda_transformed
    return LS, lda_transformed


# def svdd(k, data_matrix):
#     dt = data_matrix.transpose()
#     ddt = np.matmul(data_matrix, dt)
#     dtd = np.matmul(dt, data_matrix)
#     values_ddt, vectors_ddt = np.linalg.eig(ddt)
#     values_dtd, vectors_dtd = np.linalg.eig(dtd)
#
#     print(values_ddt.shape, vectors_ddt.shape)
#     print(values_dtd.shape, vectors_dtd.shape)
#     print(np.amax(np.array(vectors_ddt)))
#
#     eigVal = {}
#     i = 1  # create a dictionary with i as key which is tracing eigen value
#     for item in values_dtd:  # creating dictionary
#         eigVal[i] = item
#         i += 1
#     eigVal = sorted(eigVal.items(), key=lambda x: x[1], reverse=True)  # Sorting for dim reduction to choose top k
#     final_k_eigval_list = []  # stores top k key,eigenvalue pairs
#     for index, tuple in enumerate(eigVal):  # Choosing top k eigen values and storing in a list
#         if (index == k):
#             break
#         final_k_eigval_list.append(tuple)
#
#     # Extracting k eig vectors
#     m_final_eigvec_list = []
#     m_final_k_eigvec_list = []
#     n_final_eigvec_list = []
#     n_final_k_eigvec_list = []
#
#     i = 0
#     vec_transpose = vectors_dtd.transpose()
#     for tuple in eigVal:
#         m_final_eigvec_list.append(vec_transpose[tuple[0] - 1])
#         i = i + 1
#     m_final_eigvec_list = np.array(m_final_eigvec_list).transpose()
#     for i in range(len(m_final_eigvec_list)):
#         m_final_k_eigvec_list.append(m_final_eigvec_list[i][0:k])
#
#     i = 0
#     vec_transpose = vectors_ddt.transpose()
#     for tuple in eigVal:
#         n_final_eigvec_list.append(vec_transpose[tuple[0] - 1])
#         i = i + 1
#     n_final_eigvec_list = np.array(n_final_eigvec_list).transpose()
#     for i in range(len(n_final_eigvec_list)):
#         n_final_k_eigvec_list.append(n_final_eigvec_list[i][0:k])
#
#     return np.array(m_final_k_eigvec_list), np.array(n_final_k_eigvec_list), np.diag(np.array(final_k_eigval_list))


def svd(k, data_matrix):
    svd_model = TruncatedSVD(n_components=int(k))
    U = svd_model.fit_transform(data_matrix)
    VT = svd_model.components_
    VT_t = VT.transpose()
    S = np.diag(svd_model.explained_variance_)
    return VT_t, U, S


def distance_kmeans(A, B, squared=False):  # A is object matrix 40Xk(NXk) B is centorids matrix kXM
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


# def kmeans(k, X, max_iterations=300):
#     np.random.seed(42)
#     idx = np.random.choice(len(X), k, replace=False)
#     # Randomly choosing Centroids
#     centroids = X[idx, :]
#     # finding the distance between centroids and all the data points
#     distances = cdist(X, centroids, 'euclidean')
#     # Centroid with the minimum Distance
#     points = np.array([np.argmin(i) for i in distances])
#     # Repeating the above steps for a defined number of iterations
#     for _ in range(max_iterations):
#         centroids = []
#         for idx in range(k):
#             # Updating Centroids by taking mean of Cluster it belongs to
#             temp_cent = X[points == idx].mean(axis=0)
#             centroids.append(temp_cent)
#
#         centroids = np.vstack(centroids)
#         distances = cdist(X, centroids, 'euclidean')
#         points = np.array([np.argmin(i) for i in distances])
#     # latent_semantics = np.reciprocal(np.array(distance_kmeans(X,centroids) ))
#     latent_semantics = np.array(distance_kmeans(X, centroids))
#     return latent_semantics


def kmeans(k, data_matrix):
    km = KMeans(n_clusters=int(k)).fit_transform(data_matrix)
    LS = data_matrix.transpose() @ WT
    return LS, km


def save_latent_semantics(model, X, dim_red, data, data_label):
    # base_folder_path = "../Outputs/"
    # filename = "type-" + X + "_" + model + "_" + dim_red + "_" + data_label + ".csv"  # data-label stores whether it is U, V, LS
    # np.savetxt(base_folder_path + filename, data, delimiter=',')  # data stores matrix of U,V,LS..
    pass


def condense_matrix(matrix):
    condensed_matrix = []
    for i in range(0, len(matrix), 10):
        window = np.array(matrix[i: i + 10])
        condensed_matrix.append(window.mean(0))
    return np.array(condensed_matrix)


def print_to_console(sub_wt_pair_matrix, model, X):
    print("\nSubject-weight matrix for type '", X, "' and '", model, "' features: \n")
    for i in range(len(sub_wt_pair_matrix)):
        print("Subject ", i + 1, ": ", sub_wt_pair_matrix[i])

def rearrange_wt_matrix(wt_mat):
    first_9_rows = np.array(wt_mat[0])
    for i in range(1, len(wt_mat)):
        if i == 11 or i == 22 or i == 33:
            temp = wt_mat[i]
            first_9_rows = np.vstack((first_9_rows, np.array(temp)))
   
    first_9_rows = np.vstack((first_9_rows, wt_mat[35:]))
    wt_mat = np.delete(wt_mat, (0,11,22,33,35,36,37,38,39), axis=0)

    reaarranged_wt_mat = np.vstack((first_9_rows, wt_mat))
        
    return reaarranged_wt_mat


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    model, X, k, dim_red = get_sys_args()
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    result = db.find({"X": X}, {model: 1, "_id": 0})
    data_matrix = []
    if "type-" + X + "_" + model + ".csv" not in os.listdir("Data matrices"):
        for document in result:
            data_matrix.append(document[model])
        data_matrix = np.array(data_matrix)
        np.savetxt("Data matrices/type-" + X + "_" + model + ".csv", data_matrix, delimiter=',')
    else:
        data_matrix = np.loadtxt("Data matrices/type-" + X + "_" + model + ".csv", delimiter=',')
    WT = None
    LS = None
    if dim_red == "pca":
        LS, S = pca(k, data_matrix)
        WT = data_matrix @ LS
        # print(WT)
        WT = condense_matrix(WT)
        WT = rearrange_wt_matrix(WT)
        if k == 50:
            save_latent_semantics(model, X, dim_red, LS, "LS")
            save_latent_semantics(model, X, dim_red, S, "S")
            save_latent_semantics(model, X, dim_red, WT, "WT")

    elif dim_red == "svd":
        WT, VT, S = svd(k, data_matrix)
        WT = condense_matrix(WT)
        WT = rearrange_wt_matrix(WT)
        LS = VT.transpose()
        if k == 50:
            save_latent_semantics(model, X, dim_red, LS, "LS")
            save_latent_semantics(model, X, dim_red, S, "S")
            save_latent_semantics(model, X, dim_red, WT, "WT")

    elif dim_red == "lda":
        WT = lda(k, data_matrix)
        LS = data_matrix.transpose() @ WT
        WT = condense_matrix(WT)
        WT = rearrange_wt_matrix(WT)
        if k == 50:
            save_latent_semantics(model, X, dim_red, LS, "LS")
            save_latent_semantics(model, X, dim_red, WT, "WT")

    elif dim_red == "kmeans":
        WT = kmeans(k, data_matrix)
        LS = data_matrix.transpose() @ WT
        WT = condense_matrix(WT)
        WT = rearrange_wt_matrix(WT)
        if k == 50:
            save_latent_semantics(model, X, dim_red, LS, "LS")
            save_latent_semantics(model, X, dim_red, WT, "WT")

    print_to_console(WT, model, X)


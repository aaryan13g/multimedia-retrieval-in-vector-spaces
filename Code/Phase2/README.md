**Overview:**

In this phase of the project, we mainly study and implement the concepts of dimensionality curse, dimensionality reduction techniques, and page rank algorithms. We have implemented algorithms like PCA (Principal Component Analysis), SVD (Singular Value Decomposition), LDA (Latent Dirichlet Allocation), and K-Means to extract the latent semantics of a given data matrix. In addition to this, we used Personalised Page Rank and ASCOSS++ algorithms to identify the most significant subjects in the given collection of images.


**Tasks Description:**

_Task 1_:
In this task we have to implement a program in which we are given one of the three feature models, a user-specified value of X (image type), a user-specified value of k, and one of the four dimensionality reduction techniques (PCA, SVD, LDA, k-means) chosen by the user. From the given user input we have to return the top-k latent semantics extracted using images of this type. The latent semantics returned are in the form of a list of subject-weight pairs, ordered in decreasing order of weights.

_Task 2_:
In this task, we have to implement a program in which we are given one of the three feature models, a user-specified value of Y (Subject ID), a user-specified value of k, and one of the three-dimensionality reduction techniques (PCA, SVD, LDA,k-means) chosen by the user. From the given user input we have to return the top-k latent semantics extracted using images of this subject. The latent semantics returned is in the form of a list of type-weight pairs, ordered in decreasing order of weights.

_Task 3_:
In this task, we have to implement a program in which we are given one of the three feature models and a value k. Using this, we have to create and save a type-type similarity matrix and then perform a user-selected dimensionality reduction technique (PCA, SVD, LDA, k-means) on this type-type similarity matrix. We then have to report the top-k latent semantics. Each latent semantic is to be presented in the form of a list of type-weight pairs, ordered in decreasing order of weights.

_Task 4_:
In this task, we have to implement a program in which we are given one of the three feature models and a value k. Using this, we have to create and save a subject-subject similarity matrix and then perform a user-selected dimensionality reduction technique (PCA, SVD, LDA, k-means) on this subject-subject similarity matrix. We then have to report the top-k latent semantics. Each latent semantic is to be presented in the form of a list of subject-weight pairs, ordered in decreasing order of weights.

_Task 5_:
In this task, we have to implement a program in which we are given the filename of a query image that may not be in the database and a latent semantics file. Using this, we have to identify and visualize the most similar n images under the selected latent semantics.

_Task 6_:
In this task, we have to implement a program in which we are given the filename of a query image that may not be in the database and a latent semantics file. Using these inputs, we have to associate a type label (X) to the image under the selected latent semantics.

_Task 7_:
In this task, we have to implement a program in which we are given the filename of a query image that may not be in the database and a latent semantics file. Using these inputs, we have to associate a subject label (Y) to the image under the selected latent semantics.

_Task 8_:
In this task, we have to implement a program in which we are given a subject-subject similarity matrix, a value n, and a value m. Using these values we have to create a similarity graph, G(V, E), where V corresponds to the subjects in the database and E contains node pairs (vi, vj) such that, for each subject (vi, vj) we need to return the n most similar subjects in the database. The most significant m subjects in the collection have to be identified using the ASCOS++ measure.

_Task 9_:
In this task, we have to implement a program in which we are given a subject-subject similarity matrix, a value n, a value m, and three subject IDs. Using these input values we need to create a similarity graph, G(V, E), where V corresponds to the subjects in the database and E contains node pairs vi , vj such that, for each subject vi , vj we need to return the n most similar subjects in the database. The most significant m subjects (relative to the input subjects) in the collection have to be identified using the personalized PageRank measure.

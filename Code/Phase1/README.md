**Overview:**

Image retrieval is a prominent field in database systems, where we have a database and a query image, and similar images are retrieved from the database after searching. For this task, the fundamental part lies in the extraction of features of the images into vector space models and comparing and matching them with the query image through some distance/similarity metrics. The project aims to demonstrate this process through a set of tasks. Inspired by the Olivetti faces dataset, the project proceeds to extract 3 different feature descriptors for images, namely the Color Moments (CM8x8), the Extended Local Binary Patterns (ELBP), and the Histogram of Oriented Gradients (HOG). Given a folder of images, all of the above-mentioned features are extracted and stored for each image, which is then utilized to retrieve the most similar images for the given query image. This retrieval has been performed for each feature descriptor model individually, as well as for a combination of all 3 models, the matching scores have been calculated based on distance metric (Euclidean distance), and the results have been displayed in the form of the retrieved similar images.


**Task Descriptions:**

_Task 0_:
The task was to explore the Olivetti faces dataset and store the data in some format.

_Task 1_:
The task was to create 3 feature descriptor models - Color Moments (CM), Extended Local Binary Patterns (ELBP), Histogram of Gradients (HOG) for any given image.

_Task 2_:
The task was to extract and store feature descriptors for all 3 models and for all images in a given folder.

_Task 3_:
The task was to retrieve ‘K’ similar images for the given folder, image ID, and model.

_Task 4_:
This task is similar to task 3 but with a significant change that instead of using a single feature model, a combination of all models has to be used to retrieve the ‘K’ similar images and the contribution of each model is to be identified.

_Overview_:

In this project, we have developed a data retrieval pipeline for media objects that deals with vector spaces, feature extraction, dimensionality reduction, index structures, classification, and relevance feedback. The key aspects of this repository are as follows:

• Created feature extraction and dimensionality reduction models on face images

• Worked on graph analysis of vector spacesand built SVM, DT, and PageRank classifiers

• Implemented efficient search using index structures and relevance feedback


_Requirements_:

• Python 3.6+

• Windows Operating System (10 or higher)

• Visual Studio Code/PyCharm IDE


-- _Installing required Libraries_:

-Numpy - pip install numpy

-Pymongo - pip install pymongo

-Matplotlib - pip install matplotlib

-PIL -  python3 -m pip install --upgrade pip  
        python3 -m pip install --upgrade Pillow

-Sklearn - pip install scikit-image

-itertools - pip install more-itertools

Please note that some additional libraries/dependencies might be required. Please install them as per the need.


_USAGE_:

The code is written on a Python Environment. Each code file can be run individually. 

All the sample output are stored in the outputs folders(Outputs).

The code is well commented and should be self explanatory. For each and every task, the code is contained in the Code folder and the tasks can be implemented independently.

In the first phase of this project, we dealt with concepts of feature vectors where we extracted features based on models like Color moments, Extended Local Binary Patterns, and Histogram of Oriented Gradients to generate feature descriptors for images which were then used for comparing purposes using various similarity distance functions like the Euclidean distance, Manhattan Distance, etc. The code can be found under Code->Phase1 directory.

In the second phase, we delved deeper into the concepts of multimedia retrieval where we addressed the issue of the dimensionality curse. [10] Dimensionality curse is when multimedia database systems cannot manage more than a handful of facets of the multimedia data simultaneously. We were given a dataset of facial grayscale images. The images were taken during different times, varying lighting and facial expressions. We studied and implemented the dimensionality reduction techniques like PCA, SVD, LDA, and K-Means on the given image dataset and also applied page ranking algorithms to the same. The code can be found under Code->Phase2 repository.

In the final phase of this project, we study and implement classification algorithms like Support Vector Machines, Decision Trees and Personalized Page Rank. We also work with indexing algorithms like Locality-Sensitive Hashing and Vector Approximation Files to build indexing tools for efficient similar image retrieval. In addition to this we account for user feedback by building relevant feedback systems for the classification algorithms. Lastly, we build a query interface for the user to smoothly run the above-mentioned tasks. These tasks can be directly found under the Code directory.


_Interface Specification_:

This project uses the basic command line interface for taking the inputs and displaying outputs of every task. The user must have a stdin and stdout command line. The inputs are passed as command line arguments to each task as specified in the get_input or get_sys_args functions in the code files. The outputs are displayed in stdout. All the task files have to be run using python 3.6 or above.

Data Matrix created during the execution of tasks is stored in the format:
image-folder-name_feature-model.csv

Transformed Data Matrix after dimensionality reduction is stored in the format:
image-folder-name_feature-model_k_WT.csv

Latent Semantic Files created during the execution of tasks is stored in the format:
image-folder-name_feature-model_k_LS.csv


Thank You!

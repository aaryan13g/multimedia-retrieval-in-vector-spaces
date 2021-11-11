import pymongo
from pprint import pprint
import os
from PIL import Image
import numpy as np
from bson.binary import Binary
import pickle
import time
from Phase1.task1 import color_moments_model, extract_lbp, histogram_of_oriented_gradients_model

def normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Get the MongoDB client and database collection
client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
db = client.Phase2.InputFeatures

# # Next 2 lines of code to delete everything from database.
# x = db.delete_many({})
# print(x.deleted_count, " documents deleted.")

# for each image, get feature descriptors for all 3 models, and insert into the collection.
i = 0
start = time.time()
for img in os.listdir("../all/"):
    print(img)
    temp = img[:-4].split('-')
    X = temp[1]
    Y = temp[2]
    Z = temp[3]
    image_path = "../all/" + img
    image = Image.open(image_path)
    image_data = np.asarray(image) / 255   # normalize the image array
    cm = color_moments_model(image_data)
    cm = normalized(cm.flatten(order="C").tolist()).tolist()
    elbp = normalized(extract_lbp(image_data)).tolist()
    hog, _ = histogram_of_oriented_gradients_model(image_data)
    hog = normalized(hog.tolist()).tolist()
    insert_dict = {"img_name": img, "X": X, "Y": Y, "Z": Z, "cm": cm, "elbp": elbp, "hog": hog}
    db.insert_one(insert_dict)
    i = i + 1
end = time.time()
print(i, " objects inserted!")
print("Time elapsed: ", end - start, " seconds!")

# # Trial code for storing the image array in DB too as a Binary pickle dump. Ignore this!
# image_data_bin = Binary(pickle.dumps(image_data, protocol=2), subtype=128)
# data = db.find_one({'X': 'cc'})
# test = pickle.loads(data['img_array'])
# print(test)

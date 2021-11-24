import os
from numpy.lib.stride_tricks import DummyArray
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from Phase2.task5 import *

def extract_features_for_new_image(image_path):
    image = Image.open(image_path)
    image_data = np.asarray(image) / 255  # normalize the image array
    cm = color_moments_model(image_data)
    cm = normalized(cm.flatten(order="C").tolist()).tolist()
    elbp = normalized(extract_lbp(image_data)).tolist()
    hog, _ = histogram_of_oriented_gradients_model(image_data)
    hog = normalized(hog.tolist()).tolist()
    new_document = {"img_name": image_path, "cm": cm, "elbp": elbp, "hog": hog}
    return new_document

hash_table = dict()
def create_hash(results):
    for i in range(len(results)):
        hash_table.setdefault(results[i],[]).append(i+1)
    # print(hash_table)
    for keys in hash_table:
        if len(hash_table[keys])>1:
            print(hash_table[keys])
    
def decimalToBinary(n):
    return bin(n).replace("0b", "")
    
def create_VA(data_matrix,bits):
    # LSH_structure = LSH(n_layers,n_hash_per_layer,len(data_matrix[0]))
    xmax = np.amax(data_matrix,axis=0)
    # print (data_matrix)
    print("Printing!")
    print(xmax.shape)
    print("Splitting!")

    x = np.arange((pow(2,bits))+1)
    # print("X: ",x)
    l=np.true_divide(x,(pow(2,bits)))
    # print("L: ",l)
    k=0
    results=[]
   
    for i in range(len(data_matrix)):
        img=""
        for k in range(0,576):                      
        # for k in range(len(data_matrix[i])):                      
            for j in range(len(l)-1):
                if data_matrix[i][k]>=l[j] and data_matrix[i][k]<l[j+1]:
                    br=str(decimalToBinary(int(l[j]*(pow(2,bits)))))
                    while len(br)<(bits):
                        br='0'+br
                    img=img+br
            if data_matrix[i][k] == l[-1]:
                br=str(decimalToBinary(int(l[-1])))
                while len(br)<(bits):
                    br='0'+br
                img=img+br
            # print(i,k,j,br,img)
        results.append(img)  
    for i in range(0,len(results)):  
        print(i)  
        print(len(data_matrix[i]))       
        print(len(results[i]))
    print(results)
    
    print("----------------------------------------------------------------------------")
    create_hash(results)
    return results

def create_VA_Query(data_matrix,bits):
    x = np.arange((pow(2,bits))+1)
    l=np.true_divide(x,(pow(2,bits)))
    results=""  
    print(data_matrix)
    print(len(data_matrix))
    for i in range(len(data_matrix)):
        img=""                 
        for j in range(len(l)-1):
            if data_matrix[i]>=l[j] and data_matrix[i]<l[j+1]:
                br=str(decimalToBinary(int(l[j]*(pow(2,bits)))))
                while len(br)<(bits):
                    br='0'+br
                img=img+br
        if data_matrix[i] == l[-1]:
            br=str(decimalToBinary(int(l[-1])))
            while len(br)<(bits):
                br='0'+br
            img=img+br
        results=results+img
    print(results)
    print(len(results))
    return results


def get_input():
    print("Enter space-separated values of 'image','bits','folder of images':")
    image,bits, folder = input().split(" ") #Can take feature model as input too here.
    doc=extract_features_for_new_image("../images/"+image)
    image_dm=doc['elbp']
    image_dm = np.array(image_dm)
    return int(bits), folder, image_dm

def create_data_matrix(folder, feature_model):
    client = pymongo.MongoClient("mongodb+srv://mwdbuser:mwdbpassword@cluster0.u1c4q.mongodb.net/Phase2?retryWrites=true&w=majority")
    db = client.Phase2.InputFeatures
    path = '../images/' + folder + '/'
    images = [image for image in os.listdir(path)]
    data_matrix = []
    if folder + '_' + feature_model + '.csv' not in os.listdir("Data-matrices"):
        print(folder + '_' + feature_model + '.csv not found! Creating and saving it...')
        result = db.find({"img_name": {"$in": images}}, {feature_model: 1})
        i = 0
        for document in result:
            i += 1
            data_matrix.append(document[feature_model])
            print("Done: ", i)
        data_matrix = np.array(data_matrix)
        np.savetxt("Data-matrices/" + folder + '_' + feature_model + '.csv', data_matrix, delimiter=',')
    else:
        print(folder + '_' + feature_model + '.csv found!')
        data_matrix = np.loadtxt("Data-matrices/" + folder + '_' + feature_model + '.csv', delimiter=',')
        
    print("Shape of data matrix: ", data_matrix.shape)
    return data_matrix

if __name__ == "__main__":
    bits, folder ,image_dm= get_input()
    data_matrix = create_data_matrix(folder, feature_model = "elbp")
    res = create_VA(data_matrix,bits)
    img= create_VA_Query(image_dm,bits)
    

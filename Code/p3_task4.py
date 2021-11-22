import os
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

import numpy as np
    
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector):
        print("Generate_hash")
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        print("Bools ->",bools)
        hash = ''.join(bools.astype('str'))
        print("Hash ->",hash)
        return hash

    def __getitem__(self, inp_vec):
        print("Get_item_HashTable")
        hash_value = self.generate_hash(inp_vec)
        print("Hash_value", hash_value)
        return hash_value

class LSH:
    def __init__(self, n_layers, n_hash_per_layer, inp_dimensions):
        self.n_layers = n_layers
        self.hash_size = n_hash_per_layer
        self.inp_dimensions = inp_dimensions
        self.layers = list()
        self.hash_table = dict()
        for i in range(self.n_layers):
            self.layers.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def get_item(self, inp_vec,index):
        print("Get_item_LSH")
        print("INput_vector",inp_vec)
        hash_key = ""
        for table in self.layers:
            hash_key += table[inp_vec]
        print("Hash_Key",hash_key)
        #return list(set(results))
        self.hash_table.setdefault(hash_key,[]).append(index)
        #print(self.hash_table)
        return

    def get_table(self):
        return self.hash_table
    
def get_input():
    print("Enter space-separated values of 'n_layers', 'n_hash_per_layer', 'folder of images':")
    n_layers, n_hash_per_layer, folder = input().split(" ") #Can take feature model as input too here.
    return int(n_layers), int(n_hash_per_layer), folder

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

def create_LSH(data_matrix,n_layers,n_hash_per_layer):
    LSH_structure = LSH(n_layers,n_hash_per_layer,len(data_matrix[0]))
    #print(LSH_structure)
    #results = list()
    for i in range(len(data_matrix)):
        #print(image)
        #print(LSH_structure[image])
        #results.append(LSH_structure[data_matrix[i],i+1])
        LSH_structure.get_item(data_matrix[i],i+1)
        result = LSH_structure.get_table()
    return result

if __name__ == "__main__":
    n_layers, n_hash_per_layer, folder = get_input()
    data_matrix = create_data_matrix(folder, feature_model = "elbp")
    results = create_LSH(data_matrix,n_layers,n_hash_per_layer)
    print(results)

    
    
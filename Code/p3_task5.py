import os
from numpy.lib.stride_tricks import DummyArray
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from Phase2.task5 import *
from scipy.spatial import distance


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
    return hash_table

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
    print("L: ",l)
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
    return results, l

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
    print("Enter space-separated values of 'image','bits','folder of images','t':")
    image,bits, folder,t = input().split(" ") #Can take feature model as input too here.
    doc=extract_features_for_new_image("../images/"+image)
    image_dm=doc['elbp']
    image_dm = np.array(image_dm)
    return int(bits), folder, image_dm,int(t)

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

    
def get_bounds(vq,ri,p,bits):
    l = []
    imgLen = len(vq)
    res = 0
    rq= create_VA_Query(vq,bits)
    rq = [rq[i:i+bits] for i in range(0, len(rq), bits)]
    for i in range(len(rq)):
        rq[i]=int(rq[i],2)
    #rq = list(map(int, rq))
    
    
    print(p)
    print(imgLen)
    print(rq)
    print(len(ri))
    print(len(rq))
    for j in range(0,imgLen):
        if(ri[j]<rq[j]):
            l.append(vq[j] - p[ri[j]+1])
        elif(ri[j] == rq[j]):
            l.append(0)
        else:
            l.append(p[ri[j]] - vq[j])
    for i in range(0,len(l)):
        res = res+(l[i]**3)
    return res**(1/3)    

def InitCandidate(n,dst):
    for i in range(0,n):
        dst[i] = float('inf')
    return float('inf'),dst

def Candidate(d,i,n,dst,ans):
    if(d<dst[n]):
        dst[n] = d
        ans[n] = i
        temp = []
        sort_vals = np.argsort(dst)
        dst.sort()
        for i in range(0,len(sort_vals)):
            temp.append(ans[sort_vals[i]])
        ans = temp
    return dst[n],ans

def va_ssa(vq,vi,n,b):
    dst = np.zeros((n))
    count = 0
    d,dst = InitCandidate(n,dst)
    ans = np.zeros((n))
    # modData,numBits,p = va_gen(vi,b)
    
    for j in range(len(data_matrix)):
        chunks = [res[j][i:i+bits] for i in range(0, len(res[j]), bits)]
        temp=[]   
        for i in range(len(chunks)):
            temp.append(int(chunks[i],2))    
        li=get_bounds(image_dm,temp,p,bits)
        if(li<d):
            count = count+1
            d,ans = Candidate(distance.minkowski(vq,vi[i],3),i,n-1,dst,ans)
    
    
if __name__ == "__main__":
    bits, folder ,image_dm,t= get_input()
    data_matrix = create_data_matrix(folder, feature_model = "elbp")
    res,p = create_VA(data_matrix,bits)
    # img= create_VA_Query(image_dm,bits)
    hashed=create_hash(res)
    li=[]
    
    
    # for j in range(len(data_matrix)):
    #     chunks = [res[j][i:i+bits] for i in range(0, len(res[j]), bits)]
    #     temp=[]   
    #     for i in range(len(chunks)):
    #         temp.append(int(chunks[i],2))    
    #     li.append(get_bounds(image_dm,temp,p,bits))
        #chunks = list(map(int, chunks))
    
    
    # print(li)
    
    
    

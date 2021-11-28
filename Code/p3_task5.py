import os
from numpy.lib.stride_tricks import DummyArray
import pymongo
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from p3_task1 import extract_features_for_new_image, create_data_matrix
from p3_task4 import print_FP_and_miss_rates
import sys

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
    num_bins = pow(2, bits)
    x = np.arange(num_bins+1)
    l=np.true_divide(x,num_bins)
    results=[]
    for i in range(len(data_matrix)):
        img=""
        for k in range(len(data_matrix[i])):                     
            for j in range(len(l)-1):
                if data_matrix[i][k]>=l[j] and data_matrix[i][k]<l[j+1]:
                    br=str(decimalToBinary(int(l[j]*num_bins)))
                    if len(br)<(bits):
                        br = '0' * (bits - len(br)) + br
                    img=img+br
            if data_matrix[i][k] == l[-1]:
                br=str(decimalToBinary(int(l[-1])))
                if len(br)<(bits):
                    br='0' * (bits - len(br)) + br
                img=img+br
        results.append(img)  
    # for i in range(0,len(results)):  
    #     print(i)  
    #     print(len(data_matrix[i]))       
    #     print(len(results[i]))
    # print(results)
    return results, l


def get_input():
    print("Enter space-separated values of 'query image', 'bits', 'input folder', 'feature model', 't':")
    image,bits, folder, feature_model, t = input().split(" ") #Can take feature model as input too here.
    image_dm = np.array([extract_features_for_new_image("../images/" + image, feature_model)])
    return int(bits), folder, image_dm, feature_model, int(t)

    
def get_bounds(vq,ri,p,bits):
    l = []
    imgLen = len(vq[0])
    res = 0
    rq, _ = create_VA(vq,bits)
    rq = [int(rq[0][i:i+bits], 2) for i in range(0, len(rq[0]), bits)]
    # rq = list(map(int, rq))   
    # print(p)
    # print(imgLen)
    # print(rq)
    # print(len(ri))
    # print(len(rq))
    for j in range(0,imgLen):
        if(ri[j]<rq[j]):
            l.append(vq[0][j] - p[ri[j]+1])
        elif(ri[j] == rq[j]):
            l.append(0)
        else:
            l.append(p[ri[j]] - vq[0][j])
    for i in range(0,len(l)):
        res = res+(l[i]**3)
    return res**(1./3.)    

def InitCandidate(n,dst):
    for i in range(0,n):
        dst[i] = float('inf')
    return float('inf'),dst

def Candidate(d,i,n,dst,ans,count):
    
    if(d<dst[n]):
        all_nearest_images[i]=labels[int(i)]
        

        count= count +1
        dst[n] = d
        ans[n] = i
        temp = []
        sort_vals = np.argsort(dst)
        dst.sort()
        for i in range(0,len(sort_vals)):
            temp.append(ans[sort_vals[i]])
        ans = temp
    return dst[n],ans,count,all_nearest_images

def va_ssa(vq,vi,n,li,count,all_nearest_images):
    dst = np.zeros((n))
    
    d,dst = InitCandidate(n,dst)
    ans = np.zeros((n))
    rq, _ = create_VA(vq,bits)
    rq = [int(rq[0][i:i+bits], 2) for i in range(0, len(rq[0]), bits)]
    #print(rq)
    
    for j in range(len(data_matrix)):       
        if(li[j]<d):            
            d,ans,count,all_nearest_images = Candidate(distance.minkowski(rq,vi[j],3),j,n-1,dst,ans,count)
            
    return d,ans,count,all_nearest_images
    
def get_similar_images(ans):
    similar_images={}    
    k=1
    for i in ans:
        similar_images[k]=labels[int(i)]
        k=k+1
    return similar_images

def print_LSH_size(Hash_key_table):
    print("Size of LSH structure: ", sys.getsizeof(Hash_key_table), " bytes")

if __name__ == "__main__":
    bits, folder, query_image, feature_model, t= get_input()
    data_matrix, labels = create_data_matrix(folder, feature_model, label_mode="all")
    res, p = create_VA(data_matrix, bits)
    hashed=create_hash(res)
    li=[]
    temp1=[]
    for j in range(len(data_matrix)):
        chunks = [res[j][i:i+bits] for i in range(0, len(res[j]), bits)]
        temp=[]
        for i in range(len(chunks)):
            temp.append(int(chunks[i],2))
        temp1.append(temp)
        li.append(get_bounds(query_image,temp,p,bits))
        chunks = list(map(int, chunks))
    count=0 
    all_nearest_images={}
    d,ans,count,all_nearest_images=va_ssa(query_image,temp1,t,li,count,all_nearest_images)
    print(ans)
    print(d)
    # print(count)
    similar_images=get_similar_images(ans)
    print("\nThe nearest images are : \n")
    for keys in similar_images:
        print("Rank ", keys ," :" , similar_images[keys])

    print("-------------------------------------------------------------------------------------------------")
    ani={}
    i=1
    for keys in all_nearest_images:
            ani[i]=all_nearest_images[keys]
            i=i+1
            # print("Rank ", keys ," :" , all_nearest_images[keys])   

    for keys in ani:
        print("Rank ", keys ," :" , ani[keys]) 

    print_FP_and_miss_rates(similar_images,ani,query_image ,data_matrix,labels)
    print_LSH_size(hashed)
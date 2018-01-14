#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:15:29 2018

@author: wubizhi
"""
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from PIL import Image
import sys,getopt
import csv
import pickle
import glob
from scipy.spatial.distance import cosine

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model


#@profile
def npy_file():
    piam_extract_data= np.load('./piam_extract_256_256.npy')/255.0
    piam_data = np.load('./piam_256_256.npy')/255.0
    fp = open('./piam_rosette_256_name.txt')
    trained_img_names = []
    for line in fp.readlines():
        trained_img_names.append(line.split('\n')[0])
    fp.close()

    fread = open('./piam_match_gene.txt','r')
    match_DB = []
    for line in fread.readlines():
        match_DB.append(line)
    fread.close()

    with open('./AT_GO_Table.txt', 'rb') as pickle_file:
        AT_GO_Table = pickle.load(pickle_file)

    return piam_extract_data,piam_data,trained_img_names,match_DB,AT_GO_Table

def load_npy():
    piam_extract_data= np.load('./piam_tair_net_seg.npy')/255.0
    piam_data = np.load('./piam_tair_map_data.npy')/255.0
    fp = open('./piam_tair_net_seg_name.txt')
    trained_img_names = []
    for line in fp.readlines():
        trained_img_names.append(line.split('\n')[0])
    fp.close()

    fread = open('./piam_match_gene.txt','r')
    match_DB = []
    for line in fread.readlines():
        match_DB.append(line)
    fread.close()

    with open('./AT_GO_Table.txt', 'rb') as pickle_file:
        AT_GO_Table = pickle.load(pickle_file)

    return piam_extract_data,piam_data,trained_img_names,match_DB,AT_GO_Table    
#@profile
def search_GO_in_AT(AT_gene,AT_GO_Table):
    for item in AT_GO_Table:
        if(AT_gene in item ):
            return item
#@profile
def Search_Net_param_load():
    # the next part is the framework of Auoencoder for similar image retrieval
    input_img = Input(shape = (256,256,3))
    #encoder
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(input_img)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
    encoded = MaxPooling2D((2,2),padding = 'same')(x)
    #decoder
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoder = Model(input_img,encoded)
    encoder.load_weights('./encoder_400')
    autoencoder.load_weights('./autoencoder_400')
#    encoder.load_weights('./encoder_param_sgd_100')
#    autoencoder.load_weights('./autoencoder_param_sgd_100')   
    return autoencoder,encoder
#@profile
def get_top_k_similar(encoder,query_img,trained_img_encodes,trained_img_names,k):
    query_img_encoded = encoder.predict(query_img)
    query_img_encoded = np.reshape(query_img_encoded,[len(query_img_encoded),-1])
    query_cosine_array = [cosine(query_img_encoded,trained_img_encode) for trained_img_encode in trained_img_encodes]
    top_k_score = np.sort(query_cosine_array,axis = 0)[:k]
    top_k_index = np.argsort(query_cosine_array,axis = 0)[:k]
    top_k_names = [trained_img_names[index] for index in top_k_index]
    return top_k_index,top_k_names,top_k_score
#@profile
def get_top_k_gene_in_DB(top_k_names,match_DB):
    top_k_gene = []
    for name in top_k_names:
        top_k_gene_tmp = []
        for db_name in match_DB:
            db_temp = db_name.split(',')[0]
            if(name in db_temp):
                tmp = db_name.split(',')[1:]
                t_ = tmp[-1].replace('\r\n','')
                tmp[-1] = t_
                tmp.insert(0,name)
                for item in tmp:
                    if('\n' in item):
                        item=item.split('\n')[0]
                    if(item not in top_k_gene_tmp):
                        top_k_gene_tmp.append(item)
        top_ = top_k_gene_tmp[:]
        top_k_gene.append(top_)
    max_len_of_top_k_gene = 0
    for item in top_k_gene:
        if('\n' in item):
            item.remove('\n')
        if((len(item)>2) and ('None' in item)):
            item.remove('None')
        if(len(item)>max_len_of_top_k_gene):
            max_len_of_top_k_gene = len(item)
    return top_k_gene,max_len_of_top_k_gene

#@profile
def get_top_k_GO(top_k_gene,AT_GO_Table,max_len):
    top_k_GO = []
    GO_max_len = 0
    for inx in range(len(top_k_gene)):
        top_k_item = []
        top_k_item_set = set()
        if('None' not in top_k_gene[inx]):
            for ind in range(1,max_len):
                item = top_k_gene[inx][ind]
                if(item == '-'):
                    break
                else:
#                    print(item)
                    GO_str = search_GO_in_AT(item,AT_GO_Table)
                    if(GO_str==None):
                        t_ = "Not GO of "+item
                        top_k_item_set.add(t_)
#                        print(t_)
                    else:
                        GO_str_list = GO_str.split(',')
                        for i in range(1,len(GO_str_list)):
                            top_k_item_set.add(GO_str_list[i])

        item_set_len = len(top_k_item_set)
        if(item_set_len != 0):
            if(item_set_len>GO_max_len):
                GO_max_len = item_set_len
            for item in top_k_item_set:
                top_k_item.append(item)
            top_k_item = sorted(top_k_item)
        top_k_GO.append(top_k_item)

    for inx in range(len(top_k_gene)):
        for ind in range(GO_max_len - len(top_k_GO[inx])):
            top_k_GO[inx].append('-')
    return top_k_GO,GO_max_len
#@profile
def print_table(top_k_gene,max_len,score_list):

    writer_csv = csv.writer(open('table.csv','w'))

    table_title = ["gene_name"]
    for inx in range(max_len-1):
        insert_title = "latent "+str(inx+1)
        table_title.append(insert_title)
    table_title.append('score')
    table = PrettyTable(table_title)
    writer_csv.writerow(table_title)
    for inx,item in enumerate(top_k_gene):
        for ind in range(len(table_title)-len(item)-1):
            item.append("-")
        item.append(str(score_list[inx]))
        table.add_row(item)
        writer_csv.writerow(item)
    print(table)

#@profile
def print_table2(top_k_gene,max_len,score_list,top_k_names):
    table_title = ["gene_name"]
    for inx,name in enumerate(top_k_names):
        table_title.append(str(inx)+'.'+name)
    table = PrettyTable(table_title)
    score_row = ["score"]

    for score_item in score_list:
        score_row.append(score_item)
    table.add_row(score_row)

    for inx in range(1,max_len):
        new_row = ["latent gene "+str(inx)]
        for ind in range(len(top_k_gene)):
            new_row.append(top_k_gene[ind][inx])
        table.add_row(new_row)

    print(table)
#@profile
def print_table3(top_k_GO,GO_max_len,top_k_names):
    table_title = ["gene_name"]
    for inx,name in enumerate(top_k_names):
        table_title.append(str(inx)+'.'+name)
    table = PrettyTable(table_title)
    writer_csv = csv.writer(open('table3.csv','w'))
    writer_csv.writerow(table_title)
    for inx in range(GO_max_len):
        new_row = ["GO "+str(inx)]
        for ind in range(len(top_k_GO)):
            new_row.append(top_k_GO[ind][inx])
        table.add_row(new_row)
        writer_csv.writerow(new_row)
    print(table)
#@profile
def print_table4(score_list,top_k_names):
    table_title = ["gene_name","score"]
    writer_csv = csv.writer(open('table4.csv','w'))
    table = PrettyTable(table_title)
    writer_csv.writerow(table_title)
    for inx,name in enumerate(top_k_names):
        content = []
        content.append(str(inx)+'.'+name)
        content.append(score_list[inx])
        table.add_row(content)
        writer_csv.writerow(content)
    print(table)
#@profile
def piam_image_retrieve(encoder,query_img,trained_img_encodes,trained_img_names,k,testable=False):
    top_k_index,top_k_names,top_k_score = get_top_k_similar(encoder,query_img,
                                                            trained_img_encodes,trained_img_names,k)
    top_k_gene,max_len = get_top_k_gene_in_DB(top_k_names,match_DB)
    score_list = list(1 - top_k_score)
    print_table(top_k_gene,max_len,score_list)
#    print_table2(top_k_gene,max_len,score_list,top_k_names)
    top_k_GO,GO_max_len = get_top_k_GO(top_k_gene,AT_GO_Table,max_len)
    print_table3(top_k_GO,GO_max_len,top_k_names)

    fig = plt.figure(figsize=(16,12),dpi=100)
    ax=fig.add_subplot(2,k,int(k/2+1))
    plt.imshow(query_img.reshape(256,256,3))
    ax.set_title("the query image")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for k_index,n_index in enumerate(top_k_index):
        ax=fig.add_subplot(2,k,k+k_index+1)
        related_img = (piam_data[n_index]*255.0).astype('uint8')
        plt.imshow(related_img.reshape(256,256,3))
        ax.set_title(top_k_names[k_index])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.savefig('result.png',bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    import time
    opts,args = getopt.getopt(sys.argv[1:],"hi:")
    img_path  = ""
    Folder_name = 'test'
    for op,value in opts:
        if op == "-i":
            img_path = value
            start_time = time.time()
            piam_extract_data,piam_data,trained_img_names,match_DB,AT_GO_Table = npy_file()
            autoencoder,Search_Net = Search_Net_param_load()
            trained_img_encodes = np.load('tair_net_for_search_v2.npy')
            img = Image.open(img_path)
            query_img = np.reshape(img,[1,256,256,3])
            piam_image_retrieve(Search_Net,query_img,trained_img_encodes,trained_img_names,k=3,testable=False)
            print("--- %s seconds ---" % (time.time() - start_time))
        elif op == "-h":
            print("deep convolution neural network for TAIR pheotype")
            sys.exit()
#    img_path = "./piam_img_256_seg/F05420_OX_Rosette__1_pic_seg.jpg"
#    start_time = time.time()
#    piam_extract_data,piam_data,trained_img_names,match_DB,AT_GO_Table = npy_file()
#    autoencoder,Search_Net = Search_Net_param_load()
#    trained_img_encodes = np.load('feature_DB.npy')
#    img = Image.open(img_path)
#    query_img = np.reshape(img,[1,256,256,3])
#    piam_image_retrieve(Search_Net,query_img,trained_img_encodes,trained_img_names,k=3,testable=False)
#    print("--- %s seconds ---" % (time.time() - start_time))

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:13:00 2018

@author: wubizhi
"""

import TAIR_Net
import Search_Net 
import numpy as np
import sys,getopt
import glob
from skimage import color
from sklearn.cluster import KMeans
from PIL import Image
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv

def seg_macth(img,seg_img):
    lab = color.rgb2lab(seg_img)

    ab = lab[:,:,1:]
    nrows = np.shape(ab)[0]
    ncols = np.shape(ab)[1]
    ab = np.reshape(ab,[nrows*ncols,2])
    kmeans = KMeans(n_clusters=2).fit(ab)
    pixel_labels = np.reshape(kmeans.labels_,[nrows,ncols,-1])
    rgb_label = np.tile(pixel_labels,3)

    nColors = 2
    segmented_images = [[],[]]

    for k in range(nColors):
        img_color = img.copy()
        index = rgb_label != k
        img_color = np.array(img_color)
        img_color[index]=0
        segmented_images[k] = img_color

    det1 = TAIR_Net.evaluateGreen(segmented_images[0]);

    det2 = TAIR_Net.evaluateGreen(segmented_images[1]);
    det_max = max([det1,det2]);
    if(det1==det_max):
        save_img = segmented_images[0];
    else:
        save_img = segmented_images[1];
    seg_image = Image.fromarray(save_img)
    return seg_image

def get_top_k_GO(top_k_gene,AT_GO_Table,max_len):
    top_k_GO = []
    GO_max_len = 0
    search_set = set()
    self_set = set()
    ###search_set
    for inx in range(len(top_k_gene)-1):
        top_k_item = []
        top_k_item_set = set()
        if('None' not in top_k_gene[inx]):
            for ind in range(1,max_len):
                item = top_k_gene[inx][ind]
                if(item == '-'):
                    break
                else:
#                    print(item)
                    GO_str = Search_Net.search_GO_in_AT(item,AT_GO_Table)
                    if(GO_str==None):
                        t_ = "Not GO of "+item
                        top_k_item_set.add(t_)
#                        print(t_)
                    else:
                        GO_str_list = GO_str.split(',')
                        for i in range(1,len(GO_str_list)):
                            top_k_item_set.add(GO_str_list[i])
                            search_set.add(GO_str_list[i])

        item_set_len = len(top_k_item_set)
        if(item_set_len != 0):
            if(item_set_len>GO_max_len):
                GO_max_len = item_set_len
            for item in top_k_item_set:
                top_k_item.append(item)
            top_k_item = sorted(top_k_item)
        top_k_GO.append(top_k_item)
    ###self_set
    for inx in range(len(top_k_gene)-1,len(top_k_gene)):
        top_k_item = []
        top_k_item_set = set()
        if('None' not in top_k_gene[inx]):
            for ind in range(1,max_len):
                item = top_k_gene[inx][ind]
                if(item == '-'):
                    break
                else:
#                    print(item)
                    GO_str = Search_Net.search_GO_in_AT(item,AT_GO_Table)
                    if(GO_str==None):
                        t_ = "Not GO of "+item
                        top_k_item_set.add(t_)
#                        print(t_)
                    else:
                        GO_str_list = GO_str.split(',')
                        for i in range(1,len(GO_str_list)):
                            top_k_item_set.add(GO_str_list[i])
                            self_set.add(GO_str_list[i])

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
    insert_set = search_set & self_set
    share_num = len(insert_set)
    return top_k_GO,GO_max_len,share_num

def print_table(top_k_gene,max_len,score_list,img_name):

    csv_name = 'result_table/'+img_name+'_table.csv'
    writer_csv = csv.writer(open(csv_name,'w'))

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
def print_table3(top_k_GO,GO_max_len,top_k_names,img_name):
    csv_name = 'result_table/'+img_name+'_table3.csv'
    table_title = ["gene_name"]
    for inx,name in enumerate(top_k_names):
        table_title.append(str(inx)+'.'+name)
    table = PrettyTable(table_title)
    writer_csv = csv.writer(open(csv_name,'w'))
    writer_csv.writerow(table_title)
    for inx in range(GO_max_len):
        new_row = ["GO "+str(inx)]
        for ind in range(len(top_k_GO)):
            new_row.append(top_k_GO[ind][inx])
        table.add_row(new_row)
        writer_csv.writerow(new_row)
    print(table)

def piam_match(addr_list,img_name,original_img,encoder,query_img,trained_img_encodes,trained_img_names,k,testable=False):
    top_k_index,top_k_names,top_k_score = Search_Net.get_top_k_similar(encoder,query_img,
                                                            trained_img_encodes,
                                                            trained_img_names,k)
    AT_name = addr_list.split('_')[0]
    top_k_names.append(AT_name)
    temp = np.ones((len(top_k_score)+1))
    temp[0:-1] = top_k_score
    top_k_score = temp
    
    top_k_gene,max_len = Search_Net.get_top_k_gene_in_DB(top_k_names,match_DB)
    score_list = list(1 - top_k_score)
    print_table(top_k_gene,max_len,score_list,img_name)
#    print_table2(top_k_gene,max_len,score_list,top_k_names)
    top_k_GO,GO_max_len,share_num = get_top_k_GO(top_k_gene,AT_GO_Table,max_len)
    print("has the number of %d"%(share_num))
    print_table3(top_k_GO,GO_max_len,top_k_names,img_name)

    fig = plt.figure(figsize=(16,12),dpi=100)
    ax=fig.add_subplot(2,k,int(k/2+1))
    plt.imshow(original_img)
    ax.set_title("query:"+addr_list[0:-12])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for k_index,n_index in enumerate(top_k_index):
        ax=fig.add_subplot(2,k,k+k_index+1)
        related_img = (piam_data[n_index]*255.0).astype('uint8')
        plt.imshow(related_img.reshape(256,256,3))
        ax.set_title(top_k_names[k_index])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.savefig('./result_figure/'+img_name+'_result.png',bbox_inches='tight')
    return share_num

TAIR_Network,TAIR_Net_encoder = TAIR_Net.TAIR_Net_param_load()

piam_extract_data,piam_data,trained_img_names,match_DB,AT_GO_Table = Search_Net.load_npy()
autoencoder,Search_Net_encoder = Search_Net.Search_Net_param_load()
trained_img_encodes = np.load('tair_net_for_search_v2.npy')
print('load end')
addrs = glob.glob('./piam_img_test/*.jpg')
max_num = []
fp = open('img_file.txt','a')
fout = open('share_num.txt','a')
for img_path in addrs:
    print(img_path)
    img_temp = img_path[:]
    temp = img_temp.split('/')[-1]
    img_name = temp.split('.')[0]
    show_img_path = os.path.join("piam_img_256",temp[0:-8]+'.jpg')
    original_img = Image.open(show_img_path)
    img,seg_img = TAIR_Net.TAIR_Segmentation(TAIR_Network,img_path)
    seg_image = seg_macth(img,seg_img)  
    query_img = np.reshape(seg_image,[1,256,256,3])
    fp.write(img_path+'\n')
    addr_list = img_path.split('/')[-1].split('\r\n')[0]
#    AT_name = addr_list.split('_')[0]
    share_num = piam_match(addr_list,img_name,original_img,Search_Net_encoder,query_img,trained_img_encodes,trained_img_names,k=5,testable=True)
    max_num.append(share_num)
    fout.write(str(share_num)+'\n')
fp.close()
fout.close()
#import common library
import numpy as np
import sys,getopt
import glob
from skimage import color
from sklearn.cluster import KMeans
from PIL import Image
import os

#import keras library
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dropout,merge
from keras.models import Model
from keras import backend as K
from keras import optimizers

#@profile
def TAIR_Net_param_load():
    input_img = Input(shape = (256,256,3))

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    TAIR_Net = Model(input = input_img, output = conv10)
    sgd = optimizers.SGD(lr = 0.01,decay = 1e-6,momentum=0.9,nesterov = True)
    TAIR_Net.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

    TAIR_Net_encoder = Model(input_img,conv5)
    TAIR_Net_encoder.load_weights('./tair_net_encoder_30')
    TAIR_Net.load_weights('./tair_net_autoencoder_30')
    print("TAIR_Net parameter load end")
    return TAIR_Net,TAIR_Net_encoder
#@profile
def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
#@profile
def TAIR_Segmentation(TAIR_Net,img_path):
    img = Image.open(img_path)
    im2 = img.resize((256,256),Image.BICUBIC)
    query_img = np.zeros((256,256,1))
    query_img = im2.copy()
    query_img = np.reshape(query_img,[1,256,256,3])
    query_img = np.array(query_img)/255.0
    query_img = TAIR_Net.predict(query_img)
    seg_img = (query_img*255.0).astype('uint8').reshape(256,256,3)
    seg_img = Image.fromarray(seg_img)
    return im2,seg_img

#@profile
def evaluateGreen(img):
    GREEN_RANGE = [65.0/360,170/360.0]
    INTENSITY_T = 0.1
    hsv = color.rgb2hsv(img)
    relevanceMask = color.rgb2gray(img)>0
    greenAreasMask = np.logical_and((hsv[:,:,0]>GREEN_RANGE[0]),(hsv[:,:,0] < GREEN_RANGE[1]),(hsv[:,:,2] > INTENSITY_T))
    res = np.sum(greenAreasMask) / np.sum(relevanceMask);
    return res
#@profile
def seg_macth(img,seg_img,save_img_file):
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

    det1 = evaluateGreen(segmented_images[0]);

    det2 = evaluateGreen(segmented_images[1]);
    det_max = max([det1,det2]);
    if(det1==det_max):
        save_img = segmented_images[0];
    else:
        save_img = segmented_images[1];
    seg_img = Image.fromarray(save_img)
    seg_img.save(save_img_file)
    return det1+det2,det_max
#@profile
def TAIR_Net_Seg_Folder(TAIR_Net,Folder_name):
    #Folder must in the same path and this script
    #all image in this folder must be .jpg format
    addrs = glob.glob('./'+Folder_name+'/*.jpg')
    score = np.ones([len(addrs),2])
    if(not os.path.exists(Folder_name+'_seg')):
        os.mkdir(Folder_name+'_seg')
    for inx,img_path in enumerate(addrs):
        img,seg_img = TAIR_Segmentation(TAIR_Net,img_path)
        temp_name = img_path.split('/')[-1].split('.')[0]
        save_img_file = Folder_name+'_seg/'+temp_name+'_seg.jpg'
        det_sum,det_max = seg_macth(img,seg_img,save_img_file)
        score[inx][0] = det_sum
        score[inx][1] = det_max
    np.savetxt('score.txt',score,fmt='%.2f')


if __name__ == '__main__':
    opts,args = getopt.getopt(sys.argv[1:],"hi:f:")
    img_path  = ""
    Folder_name = 'test'
    for op,value in opts:
        if op == "-i":
            img_path = value
            TAIR_Net,TAIR_Net_encoder = TAIR_Net_param_load()
            memory_need = get_model_memory_usage(32,TAIR_Net)
            print(memory_need)
            img,seg_img = TAIR_Segmentation(TAIR_Net,img_path)
            #img.save(img_path)#will rewrite as size 256x256
            save_img_file = img_path[:-4]+'_seg.jpg'
            det_sum,det_max = seg_macth(img,seg_img,save_img_file)
        elif op=='-f':
            Folder_name = value
            TAIR_Net,TAIR_Net_encoder = TAIR_Net_param_load()
            TAIR_Net_Seg_Folder(TAIR_Net,Folder_name)
        elif op == "-h":
            print("deep convolution neural network for TAIR pheotype")
            sys.exit()
#    import time
#    img_path = 'barley.jpg'
#    start_time = time.time()
#    TAIR_Net,TAIR_Net_encoder = TAIR_Net_param_load()
#    memory_need = get_model_memory_usage(32,TAIR_Net)
#    print(memory_need)
#    img,seg_img = TAIR_Segmentation(TAIR_Net,img_path)
#    #img.save(img_path)#will rewrite as size 256x256
#    save_img_file = img_path[:-4]+'_seg.jpg'
#    det_sum,det_max = seg_macth(img,seg_img,save_img_file)
#    print("--- %s seconds ---" % (time.time() - start_time))
'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2021-02-21 14:55:11
LastEditTime: 2021-02-23 19:29:13
'''
import os
import sys

import pandas as pd
import pickle as pkl
import numpy as np
import cv2
from imageio import imread, imsave

image_path = "data/CROHME2016/test/"

channels=1
sentNum=0
features={}

scpFile = open('data/CROHME2016/label/test_caption_2016_normal.txt')
lines = scpFile.readlines()
for l in lines:
    line = l.strip()
    if not line: 
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '.bmp'
        im = cv2.imread(image_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

        mat = np.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8') # (1, h, w)
        
        for channel in range(channels):
            image_file = image_path + key + '.bmp'
            im = cv2.imread(image_file)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mat[channel, :, :] = im
            
        sentNum = sentNum + 1
        features[key] = mat

        print('process sentences ', sentNum)

print('load images done. sentence number ', sentNum)

outFile = 'data/test.pkl'
oupFp_feature = open(outFile, 'wb')
pkl.dump(features, oupFp_feature)

print('save file done')
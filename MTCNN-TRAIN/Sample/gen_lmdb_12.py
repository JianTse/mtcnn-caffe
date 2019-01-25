# -*- coding: utf-8 -*-
"""
	descriptor: generate mtcnn training data from source image and convert it into the lmdb database
	author: Aliang 2018-01-12
"""
import sys
import numpy as np
import cv2
import lmdb
import numpy.random as npr
import data_tran_tool
import os
import caffe
from caffe.proto import caffe_pb2
from utils import IoU
import random

anno_file = '/home/ysten/data/hand/dataSets/12/label-train.txt'

list_item = []
with open(anno_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip('\n')
        list_item.append(line)

print "total num of image: %d" % len(list_item)

stdsize = 12
lmdb_id = 0
dir_prefix = '/home/ysten/data/hand/dataSets/12/'
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
box_idx = 0
item_id = 0 # 数据库的id
batch_size = 1000 #多少图片进行一次写入,防止缓存不足

num_for_each = 1
# create the lmdb file
# map_size指的是数据库的最大容量，根据需求设置
if(lmdb_id == 0):
    lmdb_env_12 = lmdb.open(dir_prefix + 'mtcnn_train_12', map_size=1000000000)
    lmdb_txn_12 = lmdb_env_12.begin(write=True)
elif(lmdb_id == 1):
    lmdb_env_24 = lmdb.open(dir_prefix + 'mtcnn_train_24', map_size=5000000000)
    lmdb_txn_24 = lmdb_env_24.begin(write=True)
else:
    lmdb_env_48 = lmdb.open(dir_prefix + 'mtcnn_train_48', map_size=10000000000)
    lmdb_txn_48 = lmdb_env_48.begin(write=True)


# 因为caffe中经常采用datum这种数据结构存储数据
mtcnn_datum = caffe_pb2.MTCNNDatum()

rootDir = '/home/ysten/data/hand/dataSets/'
# 打乱样本顺序
random.shuffle(list_item)
for ann in list_item:
    line = ann.strip().split(' ')
    item_id += 1
    # 读入图像
    im_path = rootDir + line[0]
    img = cv2.imread(im_path)

    label = int(line[1])
    nx1 = float(line[2])
    ny1 = float(line[3])
    nx2 = float(line[4])
    ny2 = float(line[5])

    mtcnn_datum = data_tran_tool.array_to_mtcnndatum(img, label, [nx1, ny1, nx2, ny2])
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
    
    # write batch
    if(item_id) % batch_size == 0:
        if(lmdb_id == 0):
            lmdb_txn_12.commit()
            lmdb_txn_12 = lmdb_env_12.begin(write=True)
               
if (item_id+1) % batch_size != 0:
    if(lmdb_id == 0):
        lmdb_txn_12.commit()
lmdb_env_12.close()
print 'last batch'
print "There are %d images in total" % item_id
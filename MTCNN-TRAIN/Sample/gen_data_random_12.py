# -*- coding: utf-8 -*-
"""
	descriptor: generate mtcnn training data from source image and convert it into the lmdb database
	author: Aliang 2018-01-12
"""
import sys
#sys.path.append("/home/ysten/lhf/detect/MTCNN/caffe/.build_release/lib/")
import numpy as np
import cv2
import lmdb
import numpy.random as npr
import data_tran_tool
import os
import caffe
from caffe.proto import caffe_pb2
from utils import IoU

anno_file = '/home/ysten/data/hand/dataSets/train_linux.txt'

with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)

print "total num of image: %d" % num

stdsize = 12
lmdb_id = 0
dir_prefix = '/home/ysten/data/hand/dataSets/'
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

for line_idx,annotation in enumerate(annotations):

    annotation = annotation.strip().split(' ')	#每一行的数据以空白分隔符为界限
    im_path = annotation[0]					#图片的路径				
    #bbox = map(float, annotation[1:])
    bbox = [annotation[1],annotation[2],annotation[3],annotation[4]]
	
    if np.size(bbox) % 4 != 0:		#标注数据有问题
		print "the annotation data in line %d is invalid, please check file %s !" % (line_idx + 1, anno_file)
		exit(-1);
    elif np.size(bbox) == 0:
		continue;

    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    boxes_num = boxes.shape[0]
    img = cv2.imread(im_path) #读取图片
    
    if (line_idx+1) % 10 ==0:
		print line_idx + 1, "images done"

    height, width, channel = img.shape

    pos_num = 0
    part_num = 0
    neg_num = 0
    
    # neg
    while neg_num < 100:
        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            '''负样本的标签为 0'''
            item_id += 1
            
            ''' size 12'''
            if(lmdb_id == 0):
                mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im, 0, [-1.0, -1.0, -1.0, -1.0])
                keystr = '{:0>8d}'.format(item_id)
                lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString()) 
            # write batch
            if(item_id) % batch_size == 0:
                if(lmdb_id == 0):
                    lmdb_txn_12.commit()
                    lmdb_txn_12 = lmdb_env_12.begin(write=True)
            n_idx += 1
            neg_num += 1
    # pos part
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 12 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        for i in range(50):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx1) / float(size)
            offset_y2 = (y2 - ny1) / float(size)

            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                '''正样本的标签为 1'''
                item_id += 1
                ''' size 12'''
                if(lmdb_id == 0):
                    mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im, 1, [offset_x1, offset_y1, offset_x2, offset_y2])
                    keystr = '{:0>8d}'.format(item_id)
                    lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
                # write batch
                if(item_id) % batch_size == 0:
                    if(lmdb_id == 0):
                        lmdb_txn_12.commit()
                        lmdb_txn_12 = lmdb_env_12.begin(write=True)
                p_idx += 1
                pos_num += 1
            elif IoU(crop_box, box_) >= 0.4:
                '''部分样本的标签为 -1'''
                item_id += 1                
                ''' size 12'''
                if(lmdb_id == 0):
                    mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im, -1, [offset_x1, offset_y1, offset_x2, offset_y2])
                    keystr = '{:0>8d}'.format(item_id)
                    lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
                    # write batch
                    if(item_id) % batch_size == 0:
                        if(lmdb_id == 0):
                            lmdb_txn_12.commit()
                            lmdb_txn_12 = lmdb_env_12.begin(write=True)
                d_idx += 1
                part_num += 1
        box_idx += 1
        print "%s images done, pos: %s part: %s neg: %s"%(line_idx, p_idx, d_idx, n_idx)
               
if (item_id+1) % batch_size != 0:
    if(lmdb_id == 0):
        lmdb_txn_12.commit()
        lmdb_env_12.close()
    elif(lmdb_id == 1):
        lmdb_txn_24.commit()
        lmdb_env_24.close()
    elif(lmdb_id == 2):
        lmdb_txn_48.commit()
        lmdb_env_48.close()
    print 'last batch'
    print "There are %d images in total" % item_id
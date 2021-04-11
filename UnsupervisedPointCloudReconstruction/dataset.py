#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def parse_dataset_scanobject(partition,num_points=1024):
    # download('scanobjectnn')
    # DATA_DIR = tf.keras.utils.get_file(
    #     "modelnet.zip",
    #     "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    #     extract=True,
    #     cache_dir=
    # )
    # DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    DATA_DIR = '../data/ScanObjectNN'


    index = np.random.choice(2048, 1024, replace=False)

    if partition == 'train':

        f = h5py.File(os.path.join(DATA_DIR, "train.h5"))
        train_data = f['data'][:][:,index,:]
        for i in range(train_data.shape[0]):
            # mean_x, mean_y, mean_z = np.mean(train_data[i,:,0]), np.mean(train_data[i,:,1]), np.mean(train_data[i,:,2])
            train_data[i,:,0] -= (np.max(train_data[i,:,0]) + np.min(train_data[i,:,0])) / 2
            train_data[i,:,1] -= (np.max(train_data[i,:,1]) + np.min(train_data[i,:,1])) / 2
            train_data[i,:,2] -= (np.max(train_data[i,:,2]) + np.min(train_data[i,:,2])) / 2
            leng_x, leng_y, leng_z = np.max(train_data[i,:,0]) - np.min(train_data[i,:,0]), np.max(train_data[i,:,1]) - np.min(train_data[i,:,1]), np.max(train_data[i,:,2]) - np.min(train_data[i,:,2])
            if leng_x >= leng_y and leng_x >= leng_z:
                ratio = 2.0 / leng_x
            elif leng_y >= leng_x and leng_y >= leng_z:
                ratio = 2.0 / leng_y
            else:
                ratio = 2.0 / leng_z

            train_data[i,:,:] *= ratio

        train_label = f['label'][:]
        return np.array(train_data), np.array(train_label)

    elif partition == 'test':

        f = h5py.File(os.path.join(DATA_DIR, "test.h5"))
        test_data = f['data'][:][:,index,:]
        for i in range(test_data.shape[0]):
            # mean_x, mean_y, mean_z = np.mean(test_data[i,:,0]), np.mean(test_data[i,:,1]), np.mean(test_data[i,:,2])
            test_data[i,:,0] -= (np.max(test_data[i,:,0]) + np.min(test_data[i,:,0])) / 2
            test_data[i,:,1] -= (np.max(test_data[i,:,1]) + np.min(test_data[i,:,1])) / 2
            test_data[i,:,2] -= (np.max(test_data[i,:,2]) + np.min(test_data[i,:,2])) / 2
            leng_x, leng_y, leng_z = np.max(test_data[i,:,0]) - np.min(test_data[i,:,0]), np.max(test_data[i,:,1]) - np.min(test_data[i,:,1]), np.max(test_data[i,:,2]) - np.min(test_data[i,:,2])
            if leng_x >= leng_y and leng_x >= leng_z:
                ratio = 2.0 / leng_x
            elif leng_y >= leng_x and leng_y >= leng_z:
                ratio = 2.0 / leng_y
            else:
                ratio = 2.0 / leng_z

            test_data[i,:,:] *= ratio
        
        test_label = f['label'][:]
        return np.array(test_data), np.array(test_label)

def load_data_partseg(partition):
    # download('shapenetpart')
    DATA_DIR = '../data/shapenet_part_seg_hdf5_data'

    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

def parse_dataset_modelnet10(partition,num_points=1024):
    # download('modelnet10')
    # DATA_DIR = tf.keras.utils.get_file(
    #     "modelnet.zip",
    #     "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    #     extract=True,
    #     cache_dir=
    # )
    # DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    DATA_DIR = '../data/PointDA_data/modelnet'

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob(os.path.join(DATA_DIR, "[!README]*"))

    index = np.random.choice(2048, 1024, replace=False)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        if partition == 'train':
            train_files = glob(os.path.join(folder, "train/*"))
        elif partition == 'test':
            test_files = glob(os.path.join(folder, "test/*"))
        if partition == 'train':
            for f in train_files:
                raw = np.load(f)[index,:]
                # print(np.max(raw),np.min(raw))
                # mean_x, mean_y, mean_z = np.mean(raw[:,0]), np.mean(raw[:,1]), np.mean(raw[:,2])
                # raw[:,0] -= mean_x
                # raw[:,1] -= mean_y
                # raw[:,2] -= mean_z
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio
                train_points.append(raw)
                train_labels.append(i)


        elif partition == 'test':
            for f in test_files:
                raw = np.load(f)[index,:]
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio

                test_points.append(raw)
                test_labels.append(i)

    if partition == 'train':
        return np.array(train_points), np.array(train_labels)
    elif partition == 'test':
        return np.array(test_points), np.array(test_labels)


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', 
            num_points=2048, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['scanobjectnn','modelnet10', 'modelnet40', 'shapenetpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.load_name = load_name

        if dataset_name == 'modelnet40':
            self.root = os.path.join(root, dataset_name + '*hdf5_2048')
            
            self.path_h5py_all = []
            self.path_json_all = []
            if self.split in ['train','trainval','all']:   
                self.get_path('train')
            if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
                if self.split in ['val','trainval','all']: 
                    self.get_path('val')
            if self.split in ['test', 'all']:   
                self.get_path('test')

            self.path_h5py_all.sort()
            data, label = self.load_h5py(self.path_h5py_all)
            if self.load_name:
                self.path_json_all.sort()
                self.name = self.load_json(self.path_json_all)    # load label name
            
            self.data = np.concatenate(data, axis=0)
            self.label = np.concatenate(label, axis=0) 
        elif dataset_name == 'modelnet10':
            self.data, self.label = parse_dataset_modelnet10(split)
        elif dataset_name == 'scanobjectnn':
            self.data, self.label = parse_dataset_scanobject(split)
        elif dataset_name == 'shapenetpart':
            self.data, self.label, _ = load_data_partseg(split)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            # print(self.name)
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]
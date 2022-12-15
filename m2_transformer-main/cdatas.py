#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:09:37 2022

@author: sayan
"""
import os
import numpy as np
import itertools
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from torch.utils.data.dataloader import default_collate

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, precomp_data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = "/home/sayan/detmaterial/sem/MachineLearning/m2t/coco_detections.hdf5"


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        f = h5py.File(self.root_dir, 'r')

        image_id = 37209
        precomp_data = f['%d_features' % image_id][()]
        
        print("Shape = ", precomp_data.shape)
        
        delta = 50 - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:50]
        
        precomp_data = precomp_data.astype(np.float32)


        return precomp_data
    
    def collate_fn(self):
        def collate(batch):

            return default_collate(batch)

        return collate
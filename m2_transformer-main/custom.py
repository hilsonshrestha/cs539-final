#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:09:10 2022

@author: sayan
"""
import h5py
import numpy as np

import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import itertools

from cdatas import FaceLandmarksDataset

def create_batch(batch): 
    imgs       = list()

    for img in batch:
        if not img is None:
            imgs.append(img)


    if not(len(imgs) < 1): 
        bat_feats = torch.cat(imgs)
        bat_feats = bat_feats.permute((0, 3, 1, 2)) 
    return bat_feats


device = torch.device('cuda')
file = "/home/sayan/detmaterial/sem/MachineLearning/m2t/coco_detections.hdf5"

f = h5py.File(file, 'r')

image_id = 37209
precomp_data = f['%d_features' % image_id][()]

print("Shape = ", precomp_data.shape)

delta = 50 - precomp_data.shape[0]
if delta > 0:
    precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
elif delta < 0:
    precomp_data = precomp_data[:50]

precomp_data = precomp_data.astype(np.float32)

print("Shape = ", precomp_data.shape)

precomp_data = [precomp_data]

image_field = ImageDetectionsField(detections_path=file, max_detections=50, load_in_tmp=False)

p_data = FaceLandmarksDataset(precomp_data)

# Pipeline for text
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                       remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                 attention_module_kwargs={'m': 40})
decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

data = torch.load('meshed_memory_transformer.pth')
model.load_state_dict(data['state_dict'])



model.eval()
gen = {}
gts = {}


dataloader = DataLoader(p_data, batch_size=1,
                        shuffle=True, num_workers=0)


for it, (images) in enumerate(iter(dataloader)):
    images = images.to(device)
    with torch.no_grad():
        out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
    caps_gen = text_field.decode(out, join_words=False)
    print(caps_gen)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:40:28 2022

@author: wuzongze
"""

import os

import sys 
import numpy as np 
import torch

from PIL import Image
import pickle
import copy
import matplotlib.pyplot as plt
from manipulate import Manipulator

import clip


def SplitS(ds_p,M,if_std):
    all_ds=[]
    start=0
    for i in M.mindexs:
        tmp=M.dlatents[i].shape[1]
        end=start+tmp
        tmp=ds_p[start:end]
#        tmp=tmp*M.code_std[i]
        
        all_ds.append(tmp)
        start=end
    
    all_ds2=[]
    tmp_index=0
    for i in range(len(M.s_names)):
        if (not 'RGB' in M.s_names[i]) and (not len(all_ds[tmp_index])==0):
            
            if if_std:
                tmp=all_ds[tmp_index]*M.code_std[i]
            else:
                tmp=all_ds[tmp_index]
            
            all_ds2.append(tmp)
            tmp_index+=1
        else:
            tmp=np.zeros(len(M.dlatents[i][0]))
            all_ds2.append(tmp)
    return all_ds2


imagenet_templates = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(classnames, templates,model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def GetDt(classnames,model):
    text_features=zeroshot_classifier(classnames, imagenet_templates,model).t()
    
    dt=text_features[0]-text_features[1]
    dt=dt.cpu().numpy()
    

    print(np.linalg.norm(dt))
    dt=dt/np.linalg.norm(dt)
    return dt


def GetBoundary(fs3,dt,M,threshold):
    tmp=np.dot(fs3,dt)
    
    ds_imp=copy.copy(tmp)
    select=np.abs(tmp)<threshold
    num_c=np.sum(~select)


    ds_imp[select]=0
    tmp=np.abs(ds_imp).max()
    ds_imp/=tmp
    
    boundary_tmp2=SplitS(ds_imp,M,if_std=True)
    print('num of channels being manipulated:',num_c)
    return boundary_tmp2,num_c

#%%
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
    
    # pls download the checkpoint from https://drive.google.com/file/d/1FlAb1rYa0r_--Zj_ML8e6shmaF28hQb5/view
    network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/model/stylegan2-human-config-f.pkl'
    device = torch.device('cuda')
    M=Manipulator()
    M.device=device
    G=M.LoadModel(network_pkl,device)
    M.G=G
    M.SetGParameters()
    num_img=100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    np.set_printoptions(suppress=True)
    #%%
    file_path='./npy/human/'
    fs3=np.load(file_path+'fs3.npy')
    #%%
    img_indexs=np.arange(20)
    
    dlatent_tmp=[tmp[img_indexs] for tmp in M.dlatents]
    M.num_images=len(img_indexs)
    #%%

    paras=[
            ['person', 'original', 0, 0],
            ['woman', 'man', 0.2, 3],
            ['person', 'person with T-shirt', 0.15, 4],
            ['person', 'person with jeans', 0.15, 4],
            ['person', 'person with jacket', 0.15, 4],
    ]
    paras=np.array(paras)
    #%%
    
    M.step=1
    
    
    imgs=[]
    all_b=[]
    for i in range(len(paras)):
        
        neutral,target,beta,alpha=paras[i]
        beta=np.float32(beta)
        alpha=np.float32(alpha)
        M.alpha=[alpha]
        print()
        print(target)
        classnames=[target,neutral]
        dt=GetDt(classnames,model)
        boundary_tmp2,num_c=GetBoundary(fs3,dt,M,threshold=beta)
        all_b.append(boundary_tmp2)
        codes=M.MSCode(dlatent_tmp,boundary_tmp2)
        
        out=M.GenerateImg(codes)
        imgs.append(out)
    

    imgs=np.concatenate(imgs,axis=1)
    M.step=imgs.shape[1]
    M.Vis('real','',imgs,colnames=list(paras[:,1]),rownames=img_indexs,viz_size=1024)
    
    
    

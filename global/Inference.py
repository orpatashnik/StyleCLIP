#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:01:12 2021

@author: wuzongze
"""


import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")

import sys 

#sys.path=['', '/usr/local/tensorflow/avx-avx2-gpu/1.14.0/python3.7/site-packages', '/usr/local/matlab/2018b/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python37.zip', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/lib-dynload', '/usr/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/copkmeans-1.5-py3.7.egg', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/spherecluster-0.1.7-py3.7.egg', '/usr/lib/python3/dist-packages', '/usr/local/lib/python3.7/dist-packages', '/usr/lib/python3/dist-packages/IPython/extensions']

from manipulate import Manipulator
import tensorflow as tf
import numpy as np 
import torch
import clip
from PIL import Image
import pickle
import copy

import matplotlib.pyplot as plt

from MapTS import GetFs,GetBoundary,GetDt

class StyleCLIP():
    
    def __init__(self,dataset_name='ffhq'):
        print('load clip')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load("ViT-B/32", device=device)
        
#        self.LoadData('ffhq')
        self.LoadData(dataset_name)
        
    def LoadData(self, dataset_name):
        tf.keras.backend.clear_session()
        M=Manipulator(dataset_name=dataset_name)
        np.set_printoptions(suppress=True)
        
#        file_path='/cs/labs/danix/wuzongze/Tansformer_Manipulation/CLIP/results/'+M.dataset_name+'/'
#        fs3=GetFs(file_path)
        fs3=np.load('./npy/'+dataset_name+'/fs3.npy')
        
        self.M=M
        self.fs3=fs3
        
        if dataset_name=='ffhq':
            w_plus=np.load('./data/ffhq/w_plus.npy')
            self.M.dlatents=M.W2S(w_plus)
            self.c_threshold=20
        if dataset_name=='car':
            w_plus=np.load('./data/car/w_plus.npy')
            self.M.dlatents=M.W2S(w_plus)
            self.c_threshold=100
        elif dataset_name=='cat':
            self.c_threshold=100
        
        
        
        
        self.SetInitP()
    
    def SetInitP(self):
        self.M.alpha=[3]
        self.M.num_images=1
#        self.beta=0.1
        
        self.target=''
        self.neutral=''
        
#        if self.M.dataset_name=='ffhq':
#            self.target='face with long hair'
#            self.neutral='face with hair'
#        elif self.M.dataset_name=='car':
#            self.target='sports car'
#            self.neutral='car'
#        elif self.M.dataset_name=='cat':
#            self.target='cute cat'
#            self.neutral='cat'
#        elif self.M.dataset_name=='dog':
#            self.target='bulldog'
#            self.neutral='dog'
            
        
        self.GetDt2()
        
        img_index=0
        self.M.dlatent_tmp=[tmp[img_index:(img_index+1)] for tmp in self.M.dlatents]
        
        
    def GetDt2(self):
#        neutral='face with hair'
#        target='face with grey hair'
        classnames=[self.target,self.neutral]
        dt=GetDt(classnames,self.model)
        
        self.dt=dt
#        return dt
        
        num_cs=[]
        betas=np.arange(0.1,0.3,0.01)
        for i in range(len(betas)):
            boundary_tmp2,num_c=GetBoundary(self.fs3,self.dt,self.M,threshold=betas[i])
            print(betas[i])
#            print('num of channels being manipulated:',num_c, be)
            num_cs.append(num_c)
        
        num_cs=np.array(num_cs)
        select=num_cs>self.c_threshold
        
#        select_index=np.arange(len(select))[select]
        
        if sum(select)==0:
            self.beta=0.1
        else:
            self.beta=betas[select][-1]
        
    
    def GetCode(self):
        boundary_tmp2,num_c=GetBoundary(self.fs3,self.dt,self.M,threshold=self.beta)
        codes=self.M.MSCode(self.M.dlatent_tmp,boundary_tmp2)
        return codes
    
    def GetImg(self):
#        self.M.alpha=[alpha]
        
        codes=self.GetCode()
        out=self.M.GenerateImg(codes)
#        Image.fromarray(out[0,0])
        img=out[0,0]
        return img
    



#%%
if __name__ == "__main__":
    style_clip=StyleCLIP()
    self=style_clip
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

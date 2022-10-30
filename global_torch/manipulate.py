#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 21:03:58 2021

@author: wuzongze
"""


import sys

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image

import dnnlib
import legacy
import pickle
from visualizer import HtmlPageVisualizer

from torch_utils import misc
import types
from training.networks import SynthesisNetwork,SynthesisBlock,SynthesisLayer,ToRGBLayer


def change_style_code(codes, layer, channel, step):
    codes[layer][:, channel] += step
    return codes

def Vis(bname,suffix,out,rownames=None,colnames=None,save_path=None,viz_size=256):
    
    if save_path is None:
        save_path='./html/'
    
    
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=viz_size)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
    
    visualizer.save(save_path+bname+'_'+suffix+'.html')

def LoadModel(network_pkl,device):
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    G.synthesis.forward=types.MethodType(SynthesisNetwork.forward,G.synthesis)
    G.synthesis.W2S=types.MethodType(SynthesisNetwork.W2S,G.synthesis)
    
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        # print(block)
        block.forward=types.MethodType(SynthesisBlock.forward,block)
        
        if res!=4:
            layer=block.conv0
            layer.forward=types.MethodType(SynthesisLayer.forward,layer)
            layer.name='conv0_resolution_'+str(res)
            
        layer=block.conv1
        layer.forward=types.MethodType(SynthesisLayer.forward,layer)
        layer.name='conv1_resolution_'+str(res)
        
        layer=block.torgb
        layer.forward=types.MethodType(ToRGBLayer.forward,layer)
        layer.name='toRGB_resolution_'+str(res)
        
    
    return G


def S2List(encoded_styles):
    all_s=[]
    for name in encoded_styles.keys():
        tmp=encoded_styles[name].cpu().numpy()
        all_s.append(tmp)
    return all_s


# def lerp(a,b,t):
#      return a + (b - a) * t

# def Truncation(src_dlatents,dlatent_avg,truncation_psi,truncation_cutoff):
#     layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
#     ones = np.ones(layer_idx.shape, dtype=np.float32)
    
#     if truncation_cutoff is None:
#         coefs = ones*truncation_psi
#     else:
#         coefs = np.where(layer_idx > truncation_cutoff, truncation_psi * ones, ones)
#     src_dlatents_np=lerp(dlatent_avg, src_dlatents, coefs)
#     return src_dlatents_np


class Manipulator():
    def __init__(self,dataset_name='ffhq'):
#        self.file_path='./'
#        self.img_path=self.file_path+'npy/'+dataset_name+'/'
#        self.model_path=self.file_path+'model/'
#        self.dataset_name=dataset_name
#        self.model_name=dataset_name+'.pkl'
        
        self.alpha=[0] #manipulation strength 
        self.num_images=10
        self.img_index=0  #which image to start 
        # self.viz_size=256
        self.manipulate_layers=None #which layer to manipulate, list
        self.truncation_psi=0.7
        self.truncation_cutoff=8
        
#        self.G=LoadModel(self.model_path,self.model_name)
        
        self.LoadModel=LoadModel
        self.Vis=Vis
        self.S2List=S2List
        
        fmaps=[512, 512, 512, 512, 512, 256, 128,  64, 32]
        self.fmaps=np.repeat(fmaps,3)
        
    
    def GetSName(self):
        s_names=[]
        for res in self.G.synthesis.block_resolutions:
            if res==4: 
                tmp=f'conv1_resolution_{res}'
                s_names.append(tmp)
                
                tmp=f'toRGB_resolution_{res}'
                s_names.append(tmp)
            else:
                tmp=f'conv0_resolution_{res}'
                s_names.append(tmp)
                
                tmp=f'conv1_resolution_{res}'
                s_names.append(tmp)
                
                tmp=f'toRGB_resolution_{res}'
                s_names.append(tmp)
                
        return s_names
    
    def SL2D(self,tmp_code):
        encoded_styles={}
        for i in range(len(self.s_names)):
            encoded_styles[self.s_names[i]]=torch.from_numpy(tmp_code[i]).to(self.device)
        
        return encoded_styles
        
    
    
    def GenerateS(self,num_img=100):
        seed=5
        with torch.no_grad(): 
            z = torch.from_numpy(np.random.RandomState(seed).randn(num_img, self.G.z_dim)).to(self.device)
            ws = self.G.mapping(z=z,c=None,truncation_psi=self.truncation_psi,truncation_cutoff=self.truncation_cutoff)
            encoded_styles=self.G.synthesis.W2S(ws)
#            encoded_styles=encoded_styles.cpu().numpy()
            
        self.dlatents=S2List(encoded_styles)
    
    def GenerateImg(self,codes):
        
        num_images,step=codes[0].shape[:2]
        out=np.zeros((num_images,step,self.img_size,self.img_size,3),dtype='uint8')
        for i in range(num_images):
            for k in range(step):
                
                tmp_code=[]
                for m in range(len(self.s_names)):
                    tmp=codes[m][i,k][None,:]
                    tmp_code.append(tmp)
                    
                encoded_styles=self.SL2D(tmp_code)
                
                with torch.no_grad(): 
                    img = self.G.synthesis(None, encoded_styles=encoded_styles,noise_mode='const')
                    img = (img + 1) * (255/2)
                    img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                
                
                
                if img.shape[1]==img.shape[0]:
                    out[i,k,:,:,:]=img
                else:
                    tmp=img.shape[1]
                    tmp1=int((img.shape[0]-tmp)/2)
                    out[i,k,:,tmp1:tmp1+tmp,:]=img
        return out
    
    def ShowImg(self,num_img=10):
        
        codes=[]
        for i in range(len(self.dlatents)):
            # print(i)
            tmp=self.dlatents[i][:num_img,None,:]
            codes.append(tmp)
        out=self.GenerateImg(codes)
        return out
    
    def SetGParameters(self):
        self.num_layers=self.G.synthesis.num_ws
        self.img_size=self.G.synthesis.img_resolution
        self.s_names=self.GetSName()
        
        self.img_size=self.G.synthesis.block_resolutions[-1]
        
        self.mindexs=[0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21,23,24]
        
        
    
    def MSCode(self,dlatent_tmp,boundary_tmp):
        
        step=len(self.alpha)
        dlatent_tmp1=[tmp.reshape((self.num_images,-1)) for tmp in dlatent_tmp]
        dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1] # (10, 7, 512)

        l=np.array(self.alpha)
        l=l.reshape(
                    [step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
        
        if type(self.manipulate_layers)==int:
            tmp=[self.manipulate_layers]
        elif type(self.manipulate_layers)==list:
            tmp=self.manipulate_layers
        elif self.manipulate_layers is None:
            tmp=np.arange(len(boundary_tmp))
        else:
            raise ValueError('manipulate_layers is wrong')
            
        for i in tmp:
            dlatent_tmp2[i]+=l*boundary_tmp[i]
        
        codes=[]
        for i in range(len(dlatent_tmp2)):
            tmp=list(dlatent_tmp[i].shape)
            tmp.insert(1,step)
            codes.append(dlatent_tmp2[i].reshape(tmp))
        return codes
    
    
    def EditOne(self,bname,dlatent_tmp=None):
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[]
        for i in range(len(self.boundary)):
            tmp=self.boundary[i]
            if len(tmp)<=bname:
                boundary_tmp.append([])
            else:
                boundary_tmp.append(tmp[bname])
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
            
        out=self.GenerateImg(codes)
        return codes,out
    
    def EditOneC(self,cindex,dlatent_tmp=None): 
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[[] for i in range(len(self.dlatents))]
        
        #'only manipulate 1 layer and one channel'
        assert len(self.manipulate_layers)==1 
        
        ml=self.manipulate_layers[0]
        tmp=dlatent_tmp[ml].shape[1] #ada
        tmp1=np.zeros(tmp)
        tmp1[cindex]=self.code_std[ml][cindex]  #1
        boundary_tmp[ml]=tmp1
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
        out=self.GenerateImg(codes)
        return codes,out
    
    def GetFindex(self,lindex,cindex,ignore_RGB=False):
        
        if ignore_RGB:
            tmp=np.array(self.mindexs)<lindex
            tmp=np.sum(tmp)
        else:
            tmp=lindex
        findex=np.sum(self.fmaps[:tmp])+cindex
        return findex 
    
    def GetLCIndex(self,findex):
        l_p=[]
        cfmaps=np.cumsum(self.fmaps)
        for i in range(len(findex)):
            #    i=-2
            tmp_index=findex[i]
        #    importance_matrix.max(axis=0)
        #    self.attrib_indices2
            tmp=tmp_index-cfmaps
            tmp=tmp[tmp>0]
            lindex=len(tmp)
            if lindex==0:
                cindex=tmp_index
            else:
                cindex=tmp[-1]
            
            if cindex ==self.fmaps[lindex]:
                cindex=0
                lindex+=1
    #        print(completeness.index[i],completeness.iloc[i,:].values,lindex,cindex)
            l_p.append([lindex,cindex])
        l_p=np.array(l_p)
        return l_p
    def GetLCIndex2(self,findex): #input findex without ToRGB
        fmaps_o=copy.copy(self.fmaps)
        mindexs=[0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21,23,24]
        self.fmaps=fmaps_o[mindexs]
        
        l_p=self.GetLCIndex(findex)
        
        l=l_p[:,0]
        l2=np.array(mindexs)[l]
        l_p[:,0]=l2
        self.fmaps=fmaps_o
        return l_p
    
    def GetCodeMS(self):
        m=[]
        std=[]
        for i in range(len(self.dlatents)):
            tmp= self.dlatents[i] 
            tmp_mean=tmp.mean(axis=0)
            tmp_std=tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)
        
        self.code_mean=m
        self.code_std=std
        # return m,std
    
    
#%%
if __name__ == "__main__":
    # network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2_ada_pytorch/results/old/00006-joker_256-auto1-resumecustom/network-snapshot-000006.pkl'
    # save_name='018547'
    # network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2_ada_pytorch_original/results/00007-ffhq_256-mirror-auto4-noaug/network-snapshot-'+save_name+'.pkl'
    # network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/model/stylegan2-ffhq256-config-f.pkl'
    network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/model/stylegan2-human-config-f.pkl'
    device = torch.device('cuda')
#    misc.copy_params_and_buffers(G2, G, require_all=False)
    
    M=Manipulator()
    M.device=device
    G=M.LoadModel(network_pkl,device)
    M.G=G
    M.SetGParameters()
    
    #%%
    num_img=11_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    
    # num_img=100_000
    # seed=5
    # with torch.no_grad(): 
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(num_img, self.G.z_dim)).to(self.device)
    #     ws = self.G.mapping(z=z,c=None,truncation_psi=self.truncation_psi,truncation_cutoff=self.truncation_cutoff)
    #     encoded_styles=self.G.synthesis.W2S(ws)
    
    # all_s=S2List(encoded_styles)
    
    #%%
    tmp='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/results/npy/ffhq256/S_Mean_Std'
    with open(tmp, 'rb') as handle:
        _,m,std = pickle.load(handle)
    M.code_std=std[:len(M.dlatents)]
    #%%
    M.alpha=[0]
#    M.alpha=[6,3,0,-3,-6]
    # M.alpha=[24,16,8,0,-8,-16,-24]
#    M.alpha=[40,20,0,-20,-40]
    M.step=len(M.alpha)
    M.img_index=0
    M.num_images=10
    lindex,bname=6,501
#    M.
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(bname) #dlatent_tmp
    tmp=str(M.manipulate_layers)+'_'+str(bname)
    M.Vis(tmp,'c',out)
    #%%
    M.alpha=[1,-2] # -2, 1
    M.step=len(M.alpha)
    M.num_images=1
    lindex,cindex=6,501
    
    M.manipulate_layers=[lindex]
    l_path='/cs/labs/danix/wuzongze/Gan_Manipulation/pytorch-CycleGAN-and-pix2pix/datasets/smile/train/'
    
    for j in range(10_000):
        if j %1000==0:
            print(j/1000)
        M.img_index=j
        
        codes,out=M.EditOneC(cindex) #dlatent_tmp
        
        # img=Image.fromarray(out[0,0])
        
        im_AB = np.concatenate([out[0,0], out[0,1]], 1)
        img=Image.fromarray(im_AB)
        
        tmp=str(j).zfill(6)+'.jpg'
        
        tmp1=os.path.join(l_path,tmp)
        img.save(tmp1)
    
    
    
        #%%
    tmp=np.load('/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/results/npy/ffhq512/boundary_pose.npy') #boundary_pose,boundary_KidAdult
    tmp=np.tile(tmp,(1,M.G.synthesis.num_ws,1))
    tmp=torch.tensor(tmp).cuda()
    tmp1=M.G.synthesis.W2S(tmp)
    tmp1=M.S2List(tmp1)
    
    tmp2=np.zeros(tmp.shape)
    tmp2=torch.tensor(tmp2).cuda()
    tmp2=M.G.synthesis.W2S(tmp2)
    tmp2=M.S2List(tmp2)
    
    boundary_tmp=[]
    for i in range(5): #len(tmp1)
        tmp=tmp1[i]-tmp2[i]
        boundary_tmp.append(tmp)
    
    
    
    M.num_images=30
    dlatent_tmp=[tmp[M.img_index:(M.img_index+M.num_images)] for tmp in M.dlatents]
    
    
    M.alpha=np.array([-3,-2,-1,0,1,2,3])*5
    codes=M.MSCode(dlatent_tmp,boundary_tmp)
    out=M.GenerateImg(codes)
    M.Vis('tmp','c',out)
    
    #%%
    M.truncation_psi=1
    
    num_img=100_000
    M.GenerateS(num_img=num_img)
    
    out=M.ShowImg(num_img=num_img)
    
    # M.Vis('tmp','c',out)
    
    # M.G.synthesis.b4.conv1.name
    
    save_path='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2_ada_pytorch_original/results/00007-ffhq_256-mirror-auto4-noaug/img/'
    with open(save_path+'S_'+save_name, 'wb') as handle:
        pickle.dump(M.dlatents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    np.save(save_path+save_name,out[:,0])
    
    #%%
    save_name='020563'
    save_path='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2_ada_pytorch_original/results/00007-ffhq_256-mirror-auto4-noaug/img/'
    
    with open(save_path+'S_'+save_name, 'rb') as handle:
        all_s = pickle.load(handle)
    
    s_flat=np.concatenate(all_s,axis=1)
    np.save(save_path+'S_Flat_'+save_name,s_flat)
    
    #%%
    '''
    imgs=np.load(save_path+save_name+'.npy')
    M.Vis('tmp2','c',imgs[:10,None,:])
    
    with open(save_path+'S_'+save_name, 'rb') as handle:
        M.dlatents = pickle.load(handle)
    
    #%%
    
    tmp='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/results/npy/ffhq/S_mean_std'
    with open(tmp, 'rb') as handle:
        m,std = pickle.load(handle)
    M.code_std=std[:len(M.dlatents)]
    #%%
    network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/PTI/checkpoints/model_QERNCPWYXQOM_0.pt'
    with open(network_pkl, 'rb') as f_new: 
          G = torch.load(f_new).cuda()
    M.G=G
    
    #%%
    
    img_index=11
    num_layers=G.synthesis.num_ws
#    w_plus=np.load('/cs/labs/danix/wuzongze/Gan_Manipulation/encoder4editing/img_invert/out_domain/latents.npy')[img_index,:num_layers][None,:]
    w_plus=np.load('/cs/labs/danix/wuzongze/Gan_Manipulation/encoder4editing/img_invert/joker/latents.npy')[:,:num_layers]
    ws=torch.tensor(w_plus, dtype=torch.float32, device=device)
    
    with torch.no_grad(): 
        encoded_styles=G.synthesis.W2S(ws)
        M.dlatents=M.S2List(encoded_styles)
    #%%
#    M.alpha=[0]
#    M.alpha=[6,3,0,-3,-6]
    M.alpha=[24,16,8,0,-8,-16,-24]
#    M.alpha=[40,20,0,-20,-40]
    M.step=len(M.alpha)
    M.img_index=0
    M.num_images=len(ws)
    lindex,bname=9,376
#    M.
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(bname) #dlatent_tmp
    tmp=str(M.manipulate_layers)+'_'+str(bname)
    M.Vis(tmp,'c',out)
    
    
    
    
    #%%
    
    network_pkl='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/model/stylegan2-ffhq-config-f.pkl'
    with dnnlib.util.open_url(network_pkl) as fp:
        tmp = legacy.load_network_pkl(fp)
    
    tmp1='/cs/labs/danix/wuzongze/Gan_Manipulation/PTI/pretrained_models/ffhq'
    with open(tmp1, "wb") as fp:   #Pickling
        pickle.dump(tmp, fp)
    '''

















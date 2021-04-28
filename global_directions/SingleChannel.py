


import numpy as np 
import torch
import clip
from PIL import Image
import copy
from manipulate import Manipulator
import argparse

def GetImgF(out,model,preprocess):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))
    
    tmp=[]
    for i in range(len(imgs1)):
        
        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)
    
    image=torch.cat(tmp)
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    image_features1=image_features.cpu().numpy()
    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])
    
    return image_features1

def GetFs(fs):
    tmp=np.linalg.norm(fs,axis=-1)
    fs1=fs/tmp[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # 5*sigma - (-5)* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    return fs3

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='cat',
                    help='name of dataset, for example, ffhq')
    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    #%%
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    #%%
    M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    print(M.dataset_name)
    #%%
    img_sindex=0
    num_images=100
    dlatents_o=[]
    tmp=img_sindex*num_images
    for i in range(len(M.dlatents)):
        tmp1=M.dlatents[i][tmp:(tmp+num_images)]
        dlatents_o.append(tmp1)
    #%%
    
    all_f=[]
    M.alpha=[-5,5] #ffhq 5
    M.step=2
    M.num_images=num_images
    select=np.array(M.mindexs)<=16 #below or equal to 128 resolution 
    mindexs2=np.array(M.mindexs)[select]
    for lindex in mindexs2: #ignore ToRGB layers
        print(lindex)
        num_c=M.dlatents[lindex].shape[1]
        for cindex in range(num_c):
            
            M.dlatents=copy.copy(dlatents_o)
            M.dlatents[lindex][:,cindex]=M.code_mean[lindex][cindex]
            
            M.manipulate_layers=[lindex]
            codes,out=M.EditOneC(cindex) 
            image_features1=GetImgF(out,model,preprocess)
            all_f.append(image_features1)
    
    all_f=np.array(all_f)
    
    fs3=GetFs(all_f)
    
    #%%
    file_path='./npy/'+M.dataset_name+'/'
    np.save(file_path+'fs3',fs3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
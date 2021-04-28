
import os
import numpy as np
import argparse
from manipulate import Manipulator
import torch
from PIL import Image
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='ffhq',
                    help='name of dataset, for example, ffhq')

    parser.add_argument('--real', action='store_true')

    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    if not os.path.isdir('./data/'+dataset_name):
        os.system('mkdir ./data/'+dataset_name)
    #%%
    M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    print(M.dataset_name)
    #%%
    #remove all .jpg
    names=os.listdir('./data/'+dataset_name+'/')
    for name in names:
        if '.jpg' in name:
            os.system('rm ./data/'+dataset_name+'/'+name)
    
    
    #%%
    if args.real:
        latents=torch.load('./data/'+dataset_name+'/latents.pt')
        w_plus=latents.cpu().detach().numpy()
    else:
        w=np.load('./npy/'+dataset_name+'/W.npy')
        tmp=w[:50] #only use 50 images
        tmp=tmp[:,None,:]
        w_plus=np.tile(tmp,(1,M.Gs.components.synthesis.input_shape[1],1))
    np.save('./data/'+dataset_name+'/w_plus.npy',w_plus)
    
    #%%
    tmp=M.W2S(w_plus)
    M.dlatents=tmp
    
    M.img_index=0
    M.num_images=len(w_plus)
    M.alpha=[0]
    M.step=1
    lindex,bname=0,0
    
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(bname)
    #%%
    
    for i in range(len(out)):
        img=out[i,0]
        img=Image.fromarray(img)
        img.save('./data/'+dataset_name+'/'+str(i)+'.jpg')
    #%%

    
    
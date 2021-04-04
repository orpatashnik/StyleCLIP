
import os
import numpy as np
import argparse
from manipulate import Manipulator

from PIL import Image
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='ffhq',
                    help='name of dataset, for example, ffhq')

    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    if not os.path.isdir('./data/'+dataset_name):
        os.system('mkdir ./data/'+dataset_name)
    #%%
    M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    print(M.dataset_name)
    #%%
    
    M.img_index=0
    M.num_images=50
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
    w=np.load('./npy/'+dataset_name+'/W.npy')
    
    tmp=w[:M.num_images]
    tmp=tmp[:,None,:]
    tmp=np.tile(tmp,(1,M.Gs.components.synthesis.input_shape[1],1))
    
    np.save('./data/'+dataset_name+'/w_plus.npy',tmp)
    
    
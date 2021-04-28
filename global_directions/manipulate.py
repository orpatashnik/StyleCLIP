

import os
import os.path
import pickle
import numpy as np
import tensorflow as tf
from dnnlib import tflib
from global_directions.utils.visualizer import HtmlPageVisualizer


def Vis(bname,suffix,out,rownames=None,colnames=None):
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=256)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
    
    # Save results.
    visualizer.save(f'./html/'+bname+'_'+suffix+'.html')




def LoadData(img_path):
    tmp=img_path+'S'
    with open(tmp, "rb") as fp:   #Pickling
        s_names,all_s=pickle.load( fp)
    dlatents=all_s
    
    pindexs=[]
    mindexs=[]
    for i in range(len(s_names)):
        name=s_names[i]
        if not('ToRGB' in name):
            mindexs.append(i)
        else:
            pindexs.append(i)
    
    tmp=img_path+'S_mean_std'
    with open(tmp, "rb") as fp:   #Pickling
        m,std=pickle.load( fp)
    
    return dlatents,s_names,mindexs,pindexs,m,std


def LoadModel(model_path,model_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    Gs.print_layers()
    return Gs

def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])
    
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    
    np.clip(images, 0, 255, out=images)
    images=images.astype('uint8')
    return images


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        images=np.rollaxis(images, 3, 1)
    return images/ 255 *(drange[1] - drange[0])+ drange[0]


class Manipulator():
    def __init__(self,dataset_name='ffhq'):
        self.file_path='./'
        self.img_path=self.file_path+'npy/'+dataset_name+'/'
        self.model_path=self.file_path+'model/'
        self.dataset_name=dataset_name
        self.model_name=dataset_name+'.pkl'
        
        self.alpha=[0] #manipulation strength 
        self.num_images=10
        self.img_index=0  #which image to start 
        self.viz_size=256
        self.manipulate_layers=None #which layer to manipulate, list
        
        self.dlatents,self.s_names,self.mindexs,self.pindexs,self.code_mean,self.code_std=LoadData(self.img_path)
        
        self.sess=tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.Gs=LoadModel(self.model_path,self.model_name)
        self.num_layers=len(self.dlatents)
        
        self.Vis=Vis
        self.noise_constant={}
        
        for i in range(len(self.s_names)):
            tmp1=self.s_names[i].split('/')
            if not 'ToRGB' in tmp1:
                tmp1[-1]='random_normal:0'
                size=int(tmp1[1].split('x')[0])
                tmp1='/'.join(tmp1)
                tmp=(1,1,size,size)
                self.noise_constant[tmp1]=np.random.random(tmp)
        
        tmp=self.Gs.components.synthesis.input_shape[1]
        d={}
        d['G_synthesis_1/dlatents_in:0']=np.zeros([1,tmp,512])
        names=list(self.noise_constant.keys())
        tmp=tflib.run(names,d)
        for i in range(len(names)):
            self.noise_constant[names[i]]=tmp[i]
        
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.img_size=self.Gs.output_shape[-1]
    
    def GenerateImg(self,codes):
        

        num_images,step=codes[0].shape[:2]

            
        out=np.zeros((num_images,step,self.img_size,self.img_size,3),dtype='uint8')
        for i in range(num_images):
            for k in range(step):
                d={}
                for m in range(len(self.s_names)):
                    d[self.s_names[m]]=codes[m][i,k][None,:]  #need to change
                d['G_synthesis_1/4x4/Const/Shape:0']=np.array([1,18,  512], dtype=np.int32)
                d.update(self.noise_constant)
                img=tflib.run('G_synthesis_1/images_out:0', d)
                image=convert_images_to_uint8(img, nchw_to_nhwc=True)
                out[i,k,:,:,:]=image[0]
        return out
    
    
    
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
    
        
    def W2S(self,dlatent_tmp):
        
        all_s = self.sess.run(
            self.s_names,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatent_tmp})
        return all_s
        
    
    
    
    
    


#%%
if __name__ == "__main__":
    
    
    M=Manipulator(dataset_name='ffhq')
    
    
    #%%
    M.alpha=[-5,0,5]
    M.num_images=20
    lindex,cindex=6,501
    
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(cindex) #dlatent_tmp
    tmp=str(M.manipulate_layers)+'_'+str(cindex)
    M.Vis(tmp,'c',out)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





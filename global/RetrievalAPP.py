#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:20:55 2020

@author: wuzongze
"""

import sys
#sys.path=['', '/usr/local/tensorflow/avx-avx2-gpu/1.14.0/python3.7/site-packages', '/usr/local/torch/1.3/lib/python3.7/site-packages', '/usr/local/matlab/2018b/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python37.zip', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/lib-dynload', '/usr/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/copkmeans-1.5-py3.7.egg', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/spherecluster-0.1.7-py3.7.egg', '/usr/lib/python3/dist-packages', '/usr/local/lib/python3.7/dist-packages']



from tkinter import Tk,Frame ,Label,Button,Entry,PhotoImage,messagebox,Canvas
#from PIL.ImageTk import PhotoImage
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile,askopenfilename
import numpy as np

from GUI import View

import sys 

#from Inference import FullRetrieval
from Inference import StyleCLIP
import argparse
#%%

trace = 0 

class RetrievalAPP():  #Controller
    '''
    followed Model View Controller Design Pattern
    
    controller, model, view
    '''
    def __init__(self,dataset_name='ffhq'):
        
        self.root = Tk()
        self.view=View(self.root)
        self.img_ratio=2
        
#        model_path='/cs/labs/danix/wuzongze/composite_full/full/model_history/good/Open_MobileNetV2_400_2.h5' #Coco_MobileNetV2_1000_0,Open_MobileNetV2_400_2
        self.style_clip=StyleCLIP(dataset_name)
        
#        self.box=[10,20,30,40]
        
#        self.view.bg.bind('<ButtonPress-1>', self.onStart) 
#        self.view.bg.bind('<B1-Motion>',     self.onGrow)  
#        self.view.bg.bind('<Double-1>', self.open_img)
#        self.view.bg.bind('<ButtonRelease-1>', self.MakeHole)
#        self.view.set_p.bind('<ButtonPress-1>', self.SetParameters) 
#        self.view.rset_p.bind('<ButtonPress-1>', self.ResetParameters) 
#        self.view.set_p.command= self.SetParameters
        
        
#        self.view.set_category.bind("<<ComboboxSelected>>", self.ChangeDataset)
        self.view.neutral.bind("<Return>", self.text_n)
        self.view.target.bind("<Return>", self.text_t)
        self.view.alpha.bind('<ButtonRelease-1>', self.ChangeAlpha)
        self.view.beta.bind('<ButtonRelease-1>', self.ChangeBeta)
        self.view.set_init.bind('<ButtonPress-1>', self.SetInit) 
        self.view.reset.bind('<ButtonPress-1>', self.Reset) 
        self.view.bg.bind('<Double-1>', self.open_img)
        
        
        self.drawn  = None
        
        self.view.target.delete(1.0, "end")
        self.view.target.insert("end", self.style_clip.target)
#        
        self.view.neutral.delete(1.0, "end")
        self.view.neutral.insert("end", self.style_clip.neutral)
        
    
    def Reset(self,event):
        self.style_clip.GetDt2()
        self.style_clip.M.alpha=[0]
        
        self.view.beta.set(self.style_clip.beta)
        self.view.alpha.set(0)
        
        img=self.style_clip.GetImg()
        img=Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.addImage_m(img)
        
    
    def SetInit(self,event):
        codes=self.style_clip.GetCode()
        self.style_clip.M.dlatent_tmp=[tmp[:,0] for tmp in codes]
        print('set init')
    
    def ChangeAlpha(self,event):
        tmp=self.view.alpha.get()
        self.style_clip.M.alpha=[float(tmp)]
        
        img=self.style_clip.GetImg()
        print('manipulate one')
        img=Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.addImage_m(img)
        
    def ChangeBeta(self,event):
        tmp=self.view.beta.get()
        self.style_clip.beta=float(tmp)
        
        img=self.style_clip.GetImg()
        print('manipulate one')
        img=Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.addImage_m(img)

    def ChangeDataset(self,event):
        
        dataset_name=self.view.set_category.get()
        
        self.style_clip.LoadData(dataset_name)
        
        self.view.target.delete(1.0, "end")
        self.view.target.insert("end", self.style_clip.target)
        
        self.view.neutral.delete(1.0, "end")
        self.view.neutral.insert("end", self.style_clip.neutral)
    
    def text_t(self,event):
        tmp=self.view.target.get("1.0",'end')
        tmp=tmp.replace('\n','')
        
        self.view.target.delete(1.0, "end")
        self.view.target.insert("end", tmp)
        
        print('target',tmp,'###')
#        print('###')
        self.style_clip.target=tmp
        self.style_clip.GetDt2()
        self.view.beta.set(self.style_clip.beta)
        self.view.alpha.set(3)
        self.style_clip.M.alpha=[3]
        
        img=self.style_clip.GetImg()
        print('manipulate one')
        img=Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.addImage_m(img)
        
        
    def text_n(self,event):
        tmp=self.view.neutral.get("1.0",'end')
        tmp=tmp.replace('\n','')
        
        self.view.neutral.delete(1.0, "end")
        self.view.neutral.insert("end", tmp)
        
        print('neutral',tmp,'###')
#        print('###')
        self.style_clip.neutral=tmp
#        self.style_clip.GetDt2()
        self.view.target.delete(1.0, "end")
        self.view.target.insert("end", tmp)
        
        
    
    def run(self):
        self.root.mainloop()
        
        

    
    def addImage(self,img):
        self.view.bg.create_image(self.view.width/2, self.view.height/2, image=img, anchor='center')
        self.image=img #save a copy of image. if not the image will disappear
        
#        label = tk.Label(frame_in, image=self.images[type_img])
#        label.pack(side="right")
        
        
    def addImage_m(self,img):
        self.view.mani.create_image(512, 512, image=img, anchor='center')
        self.image2=img
        

        
        
    
    
    def openfn(self):
#        filename = askopenfilename(title='open',initialdir='/tmp/data2/'+self.style_clip.M.dataset_name+'/',filetypes=[("all image format", ".jpg")])
        filename = askopenfilename(title='open',initialdir='./data/'+self.style_clip.M.dataset_name+'/',filetypes=[("all image format", ".jpg"),("all image format", ".png")])
#        filename = askopenfilename(title='open',filetypes=[("all image format", ".jpg")])
#        filename = askopenfilename(title='open',initialdir='./data/ffhq/',filetypes=[("all image format", ".jpg"),])
        return filename
    
    def open_img(self,event):
        x = self.openfn()
        print(x)
        
        
        img = Image.open(x)
        img2 = img.resize(( 512,512), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img2)
        self.addImage(img2)
        
        img = ImageTk.PhotoImage(img)
        self.addImage_m(img)
        
        img_index=x.split('/')[-1].split('.')[0]
        img_index=int(img_index)
        print(img_index)
        self.style_clip.M.img_index=img_index
        self.style_clip.M.dlatent_tmp=[tmp[img_index:(img_index+1)] for tmp in self.style_clip.M.dlatents]
        
        
        self.style_clip.GetDt2()
        self.view.beta.set(self.style_clip.beta)
        self.view.alpha.set(3)
        
        
        
#        return img
    #%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='ffhq',
                    help='name of dataset, for example, ffhq')
    
    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    self=RetrievalAPP(dataset_name)
    self.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
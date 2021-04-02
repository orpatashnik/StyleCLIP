#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:25:42 2020

@author: wuzongze
"""

import sys
#sys.path=['', '/usr/local/tensorflow/avx-avx2-gpu/2.0.0/python3.7/site-packages',
#          '/usr/local/torch/1.3/lib/python3.7/site-packages',
#          '/usr/local/matlab/2018b/lib/python3.7/site-packages', 
#          '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python37.zip', 
#          '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7', 
#          '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/lib-dynload',
#          '/usr/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages',
#          '/usr/local/lib/python3.7/dist-packages', '/usr/lib/python3/dist-packages']

#sys.path=['', '/usr/local/tensorflow/avx-avx2-gpu/1.14.0/python3.7/site-packages', '/usr/local/matlab/2018b/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python37.zip', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/lib-dynload', '/usr/lib/python3.7', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/copkmeans-1.5-py3.7.egg', '/cs/labs/danix/wuzongze/pythonV/venv3.7/lib/python3.7/site-packages/spherecluster-0.1.7-py3.7.egg', '/usr/lib/python3/dist-packages', '/usr/local/lib/python3.7/dist-packages']


from tkinter import Tk,Frame ,Label,Button,Entry,PhotoImage,messagebox,Canvas,Text,Scale
#from tkinter.tkFileDialog import askopenfile
from tkinter.filedialog import askopenfile,askopenfilename
from PIL import  Image
#from PIL import ImageTk

#from GUI_bg import CanvasEventsDemo

#import PIL
#from PIL import ImageTk
#from PIL import Image
from tkinter import  ttk,HORIZONTAL



class View():
    def __init__(self,master):
        
        self.width=600
        self.height=600
        
        
        self.root=master
        self.root.geometry("600x600")
#        self.root.bind('<ButtonPress-1>', self.TestEvent) 
        
        self.left_frame=Frame(self.root,width=600)
        self.left_frame.pack_propagate(0)
        self.left_frame.pack(fill='both', side='left', expand='True')
        
        self.retrieval_frame=Frame(self.root,bg='snow3')
        self.retrieval_frame.pack_propagate(0)
        self.retrieval_frame.pack(fill='both', side='right', expand='True')
        
        self.bg_frame=Frame(self.left_frame,bg='snow3',height=600,width=600)
        self.bg_frame.pack_propagate(0)
        self.bg_frame.pack(fill='both', side='top', expand='True')
#        self.bg_frame.grid(row=0, column=0,padx=30, pady=30)
        
        self.command_frame=Frame(self.left_frame,bg='snow3')
        self.command_frame.pack_propagate(0)
        self.command_frame.pack(fill='both', side='bottom', expand='True')
#        self.command_frame.grid(row=1, column=0,padx=0, pady=0)
        
        self.bg=Canvas(self.bg_frame,width=self.width,height=self.height, bg='gray')
        self.bg.place(relx=0.5, rely=0.5, anchor='center')
        
        self.mani=Canvas(self.retrieval_frame,width=1024,height=1024, bg='gray') 
        self.mani.grid(row=0, column=0,padx=0, pady=42)
        
        
#        print(self.bg.start.x,self.bg.start.y)
        self.SetCommand()
        
        
        
    
    def run(self):
        self.root.mainloop()
    
#    def TestEvent(self,event):
#        print(event.widget,event.x,event.y)
    
    def helloCallBack(self):
        category=self.set_category.get()
#        print('####',category)
#        box=self.bg.box
#        messagebox.showinfo( "Hello Python",str(box[0]))
        messagebox.showinfo( "Hello Python",category)
    
        
        
    
    def SetCommand(self):
        
#        tmp = Label(self.command_frame, text="dataset", width=10,bg='snow3')
#        tmp.grid(row=0, column=0,padx=0, pady=0)
#        names=['None','ffhq','car','cat']
#        self.set_category = ttk.Combobox(self.command_frame, 
#                            values=names,width=10)
#        self.set_category.current(0)
#        self.set_category.grid(row=0, column=2, padx=10, pady=10)
        
        tmp = Label(self.command_frame, text="neutral", width=10 ,bg='snow3')
        tmp.grid(row=1, column=0,padx=10, pady=10)
        
        tmp = Label(self.command_frame, text="a photo of a", width=10 ,bg='snow3')
        tmp.grid(row=1, column=1,padx=10, pady=10)
        
        self.neutral = Text ( self.command_frame, height=2, width=30)
        self.neutral.grid(row=1, column=2,padx=10, pady=10)
        
        
        tmp = Label(self.command_frame, text="target", width=10 ,bg='snow3')
        tmp.grid(row=2, column=0,padx=10, pady=10)
        
        tmp = Label(self.command_frame, text="a photo of a", width=10 ,bg='snow3')
        tmp.grid(row=2, column=1,padx=10, pady=10)
        
        self.target = Text ( self.command_frame, height=2, width=30)
        self.target.grid(row=2, column=2,padx=10, pady=10)
        

        
#        self.set_p = Button(self.command_frame, text="Set Parameters")#,command= self.helloCallBack) 
#        self.set_p.grid(row=2, column=3, padx=10, pady=10)
        
        
        tmp = Label(self.command_frame, text="strength", width=10 ,bg='snow3')
        tmp.grid(row=3, column=0,padx=10, pady=10)
        
        self.alpha = Scale(self.command_frame, from_=-15, to=25, orient=HORIZONTAL,bg='snow3', length=250,resolution=0.01)
        self.alpha.grid(row=3, column=2,padx=10, pady=10)
        
        
        tmp = Label(self.command_frame, text="disentangle", width=10 ,bg='snow3')
        tmp.grid(row=4, column=0,padx=10, pady=10)
        
        self.beta = Scale(self.command_frame, from_=0.08, to=0.4, orient=HORIZONTAL,bg='snow3', length=250,resolution=0.001)
        self.beta.grid(row=4, column=2,padx=10, pady=10)
        
        self.reset = Button(self.command_frame, text='Reset') 
        self.reset.grid(row=5, column=1,padx=10, pady=10)
        
        
        self.set_init = Button(self.command_frame, text='Accept') 
        self.set_init.grid(row=5, column=2,padx=10, pady=10)
    
#        self.set_p = Button(self.command_frame, text="Set Parameters")#,command= self.helloCallBack) 
#        self.set_p.grid(row=3, column=0, padx=10, pady=10)
#        
#        self.rset_p = Button(self.command_frame, text="Reset Parameters")#,command= self.helloCallBack) 
#        self.rset_p.grid(row=3, column=1, padx=10, pady=10)

#%%
if __name__ == "__main__":
    master=Tk()
    self=View(master)
    self.run()
    
    
    
    
    
    
    
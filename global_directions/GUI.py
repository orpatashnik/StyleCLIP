

from tkinter import Tk,Frame ,Label,Button,messagebox,Canvas,Text,Scale
from tkinter import  HORIZONTAL

class View():
    def __init__(self,master):
        
        self.width=600
        self.height=600
        
        
        self.root=master
        self.root.geometry("600x600")
        
        self.left_frame=Frame(self.root,width=600)
        self.left_frame.pack_propagate(0)
        self.left_frame.pack(fill='both', side='left', expand='True')
        
        self.retrieval_frame=Frame(self.root,bg='snow3')
        self.retrieval_frame.pack_propagate(0)
        self.retrieval_frame.pack(fill='both', side='right', expand='True')
        
        self.bg_frame=Frame(self.left_frame,bg='snow3',height=600,width=600)
        self.bg_frame.pack_propagate(0)
        self.bg_frame.pack(fill='both', side='top', expand='True')
        
        self.command_frame=Frame(self.left_frame,bg='snow3')
        self.command_frame.pack_propagate(0)
        self.command_frame.pack(fill='both', side='bottom', expand='True')
#        self.command_frame.grid(row=1, column=0,padx=0, pady=0)
        
        self.bg=Canvas(self.bg_frame,width=self.width,height=self.height, bg='gray')
        self.bg.place(relx=0.5, rely=0.5, anchor='center')
        
        self.mani=Canvas(self.retrieval_frame,width=1024,height=1024, bg='gray') 
        self.mani.grid(row=0, column=0,padx=0, pady=42)
        
        self.SetCommand()
        
        
        
    
    def run(self):
        self.root.mainloop()
    
    def helloCallBack(self):
        category=self.set_category.get()
        messagebox.showinfo( "Hello Python",category)
    
    def SetCommand(self):
        
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

#%%
if __name__ == "__main__":
    master=Tk()
    self=View(master)
    self.run()
    
    
    
    
    
    
    
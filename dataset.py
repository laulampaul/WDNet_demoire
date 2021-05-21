import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import random

def default_loader(path1,path2):
    #pdb.set_trace()
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w,h=img1.size
    '''
    x=random.randint(0,w-256)
    y=random.randint(0,h-256)
    #iConv')!=-1:
    img1=img1.crop((x,y,x+256,y+256))
    img2=img2.crop((x,y,x+256,y+256))
    '''
    #print(path1,path2)
    
    assert(path1.split('_')[-2]==path2.split('_')[-2])
    # demoire photo dataset
    i = random.randint(-6,6)
    j = random.randint(-6,6)
    img1=img1.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j))
    img2=img2.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j))
    
    #img1 = img1[int(w/4):int(3*w/4),int(h/4):int(h*3/4)]
    #img2 = img2[int(w/4):int(3*w/4),int(h/4):int(h*3/4)]
    img1 = img1.resize((256,256),Image.BILINEAR )
    img2 = img2.resize((256,256),Image.BILINEAR )
    
    r= random.randint(0,1)
    if r==1:
        img1=img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2=img2.transpose(Image.FLIP_LEFT_RIGHT)
        
    t = random.randint(0,2)
    if t==0:
        img1=img1.transpose(Image.ROTATE_90)
        img2=img2.transpose(Image.ROTATE_90)
    elif t==1:
        img1=img1.transpose(Image.ROTATE_180)
        img2=img2.transpose(Image.ROTATE_180)
    elif t==2:
        img1=img1.transpose(Image.ROTATE_270)
        img2=img2.transpose(Image.ROTATE_270)
    '''
    x=random.randint(0,w-256)
    y=random.randint(0,h-256)
    #iConv')!=-1:
    img1=img1.crop((x,y,x+256,y+256))
    img2=img2.crop((x,y,x+256,y+256))
    '''
    
    '''
    minnum= min(w,h);
    if minnum>256:
    	k=random.randint(0,minnum-256)
    	img1=img1.crop((k,k,k+256,k+256))
    	img2=img2.crop((k,k,k+256,k+256))
    else:
      img1=img1.resize((256,256));
      img2=img1.resize((256,256));
    #print(img1.size)
    '''
    return img1 ,img2
    

class myImageFloder(data.Dataset):
    def __init__(self,root,transform = None,target_transform = None,loader = default_loader):


        #c = 0
        imgin = []
        imgout = []
        imgin_names = []
        imgout_names = []

        for img_name in os.listdir(os.path.join(root,'source')):
            if img_name !='.' and img_name !='..':
                imgin_names.append(os.path.join(root,'source',img_name))
                
        for img_name in os.listdir(os.path.join(root,'target')):
            if img_name !='.' and img_name !='..':
                imgout_names.append(os.path.join(root,'target',img_name))
        imgin_names.sort()
        imgout_names.sort()
        #imgin_names = imgin_names[0:104814]
        #imgout_names = imgout_names[0:104814]

        print(len(imgin_names),len(imgout_names))

        assert len(imgin_names)==len(imgout_names)
        self.root = root
        self.imgin_names = imgin_names
        self.imgout_names = imgout_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self,index):
        imgin = self.imgin_names[index]
        imgout = self.imgout_names[index]
        
        img1,img2 = self.loader(imgin,imgout)
        
        if self.transform is not None:
            #pdb.set_trace()
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2

    def __len__(self):
        return len(self.imgin_names)

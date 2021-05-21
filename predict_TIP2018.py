import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.models.vgg as vgg
#from model_demoiregan2 import *
#from model_pixpix import *
#from model_partialcov import *
#from model_newunet import *
from model_dense import *
import pdb
from torchvision import transforms
from skimage import measure
#from pixtopix import LossNetwork
from skimage import color
criterion_GAN = torch.nn.MSELoss()

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform1 = transforms.Compose([
      transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ]
    )
    
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.double) - im2.astype(np.double)) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr   

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",        #1_2 to 5_2
        }
        
    def forward(self, x):
        output = {}
        #import pdb
        #pdb.set_trace()
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #import pdb
        #pdb.set_trace()
        return output
        
transform1 = transforms.Compose([
      transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ])       
def predict_img(net,
                #cls,
                img,
                lossnet,
                use_gpu=False):
    #pdb.set_trace()
    
    
    transform1 = transforms.Compose([
      transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ]
    )
    img=transform1(img)
    #x_r = (real_A[:,0,:,:]*255-105.648186)/255.+0.5
    #x_g = (real_A[:,1,:,:]*255-95.4836)/255.+0.5
    #x_b = (real_A[:,2,:,:]*255-86.4105)/255.+0.5
    # = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
    
    #    y_r = (real_B[:,0,:,:]*255-121.2556)/255.+0.5
    #    y_g = (real_B[:,1,:,:]*255-114.89969)/255.+0.5
    #    y_b = (real_B[:,2,:,:]*255-102.02478)/255.+0.5
    #    real_B = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)

    net.eval()
    lossnet.eval()
    
    w,h = img.shape[1],img.shape[2]
    tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1)))
    real_a_pre = lossnet((img*255-tensor_c).cuda()) 
    
    relu_1 = nn.functional.interpolate(real_a_pre['relu1'].detach(),size=(w,h))
    relu_2 = nn.functional.interpolate(real_a_pre['relu2'].detach(),size=(w,h))
    relu_3 = nn.functional.interpolate(real_a_pre['relu3'].detach(),size=(w,h))
    '''
    img.shape
    tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1)))
    real_a_pre = lossnet((img*255-tensor_c).cuda()) 
    
    relu_1 = nn.functional.interpolate(real_a_pre['relu1'].detach(),size=(256,256))
    relu_2 = nn.functional.interpolate(real_a_pre['relu2'].detach(),size=(256,256))
    relu_3 = nn.functional.interpolate(real_a_pre['relu3'].detach(),size=(256,256))
    relu_4 = nn.functional.interpolate(real_a_pre['relu4'].detach(),size=(256,256))
    relu_5 = nn.functional.interpolate(real_a_pre['relu5'].detach(),size=(256,256))
    '''     
    precept = torch.cat([relu_1/255.,relu_2/255.,relu_3/255.],1)#,relu_4/255.,relu_5/255.], 1)
    
    img=img.unsqueeze(0)
    '''
    x_r = (img[:,0,:,:]*255-105.648186)/255.+0.5
    x_g = (img[:,1,:,:]*255-95.4836)/255.+0.5
    x_b = (img[:,2,:,:]*255-86.4105)/255.+0.5
    img = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
    
    x_r = (img[:,0,:,:]*255-120.497406)/255.+0.5
    x_g = (img[:,1,:,:]*255-114.58455)/255.+0.5
    x_b = (img[:,2,:,:]*255-102.13702)/255.+0.5
    img = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
    '''
    x_r = (img[:,0,:,:]*255-105.648186)/255.+0.5
    x_g = (img[:,1,:,:]*255-95.4836)/255.+0.5
    x_b = (img[:,2,:,:]*255-86.4105)/255.+0.5
    img = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
  
    y_r = ((img[:,0,:,:]-0.5)*255+121.2556)/255.
    y_g = ((img[:,1,:,:]-0.5)*255+114.89969)/255.
    y_b = ((img[:,2,:,:]-0.5)*255+102.02478)/255.
    img = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
    if use_gpu:
        img = img.cuda()
        net = net.cuda()
        
    with torch.no_grad():
        #pdb.set_trace()
        #_,c,w,h=img.shape
        #mask_predict = cls(img)
        #mask_p = mask_predict[:,0,:,:] +mask_predict[:,1,:,:] +mask_predict[:,2,:,:]
        #filters = Variable(torch.ones(1,1,16,16)).cuda()
        #mask_p = mask_p.unsqueeze(0)
            #mask_p = mask_p.cuda()
        '''    #pdb.set_trace()
        mask_p = F.conv2d(mask_p,filters,stride = 16)
            #pdb.set_trace()
        mask_p = mask_p/255.*0.2
        _,c,w,h=mask_p.shape

        mask_p = mask_p.view(1,1,-1,w*h)
        aa,bb = torch.topk(mask_p,8,largest=False)            # how many patch need to search:10

        for one in range(8):
            mask_p[0,0,0,bb[0,0,0,one]]=1

        mask_p[mask_p<0.999 ]=0
        mask_p[mask_p>1.0001 ]=0

        mask_p = mask_p.view(1,1,w,h)
        '''
        '''
        mask_predict = cls(img)
        mask_p = mask_predict.view(1,1,-1,256)
        aa,bb = torch.topk(mask_p,10,largest=False)            # how many patch need to search:10

        for one in range(10):
            mask_p[0,0,0,bb[0,0,0,one]]=1

        mask_p[mask_p<0.999]=0
        mask_p[mask_p>1.0001]=0
        mask_p = mask_p.view(1,1,16,16)
        '''
        imgin = wavelet_dec(img)
        imgout = net(Variable(imgin))
        imgout =wavelet_rec(imgout) + img
        #imgout=imgout+img
        imgout = imgout.squeeze(0)
        #x_r = ((imgout[0,:,:]-0.5)*255+120.497406)/255.
        #x_g = ((imgout[1,:,:]-0.5)*255+114.58455)/255.
        #x_b = ((imgout[2,:,:]-0.5)*255+102.13702)/255.
        #imggt = torch.cat([ x_r.unsqueeze(0) ,x_g.unsqueeze(0) ,x_b.unsqueeze(0)  ],0)

    
    return imgout

crit = criterion_GAN.cuda() 

def get_args():
    parser = argparse.ArgumentParser()             #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--model', '-m', default='./saved_models/facades3/lastest.pth',#'/data1/liul/model/generatorsole7_100.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--cmodel', '-x', default='./saved_models/facades2/classfier.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='file floder names of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)


    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()

if __name__ == "__main__":
    args = get_args()
    in_files = 'input'#args.input
    in_files2 = 'output'
    out_files = get_output_filenames(args)

    net = GeneratorUNet()
    lossnet= LossNetwork()
    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)        
    #classfier =  Discriminator2()
    #net = torch.nn.DataParallel(net)
    #net.apply(weights_init)

    
    print("Begin Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net = nn.DataParallel(net)

        net.load_state_dict(torch.load(args.model))
        lossnet.cuda()
        wavelet_dec.cuda()
        wavelet_rec.cuda()
        #classfier.cuda()
        #classfier.load_state_dict(torch.load(args.cmodel))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        #classfier.cpu()
        #classfier.load_state_dict(torch.load(args.cmodel, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")
    #pdb.set_trace()
    root = '../dataset10w/test'
    #net = nn.DataParallel(net)
    im_files=os.listdir(os.path.join(root,in_files))  #/data2/liul/test
    im_files2=os.listdir(os.path.join(root,in_files2))
    im_files.sort()
    im_files2.sort()
    #im_files = im_files[0:100]
    #im_files2 = im_files2[0:100]
    #pdb.set_trace()
    log=open('result.txt','w')
    psnr_ori=0
    psnr_pro=0
    ssim_ori=0
    ssim_pro=0
    for i, fn in enumerate(im_files):
        
        print("\nprocessing image {} ...".format(fn))

        img = Image.open(os.path.join(root,in_files,fn))
        #pdb.set_trace()
        [w,h]=img.size
        img=img.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
        #imgcom=img.resize((256, 256),Image.ANTIALIAS)
        #imgcom = img
        img = img.resize((256, 256),Image.BILINEAR)
        imgcom = img
        imggt = Image.open(os.path.join(root,in_files2,im_files2[i]))
        imggt=imggt.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
        
        #[w,h]=imggt.size
        #imggt=imggt.crop((int(w/4),int(h/4),int(w*3/4),int(h*3/4)))
        imggt=imggt.resize((256,256),Image.BILINEAR)
        imggt = transform1(imggt)
        #pdb.set_trace()
        #x_r = (imggt[0,:,:]*255-120.497406)/255.+0.5
        #x_g = (imggt[1,:,:]*255-114.58455)/255.+0.5
        #x_b = (imggt[2,:,:]*255-102.13702)/255.+0.5
        #imggt = torch.cat([ x_r.unsqueeze(0) ,x_g.unsqueeze(0) ,x_b.unsqueeze(0)  ],0)
        #120.497406, 114.58455,  102.13702
        if img is None:
            continue;

        
        img2 = predict_img(net=net,
                           #cls=classfier,
                           img=imgcom,
                           lossnet=lossnet,
                           use_gpu=not args.cpu
                           )
        #pdb.set_trace()
        mse = crit(img2,imggt.cuda())
        #print(mse)
        
        img2 = (img2)*255
        #pdb.set_trace()
        #img2=img2*255
        img2=torch.clamp(img2,0,255);
        img2= np.uint8(img2.data.cpu().numpy())
        img2= img2.transpose((1,2,0))
        
        img2= Image.fromarray(img2)
        
        imggt = (imggt)*255
        imggt=torch.clamp(imggt,0,255);
        
        
        #pdb.set_trace()
        imggt= np.uint8(imggt.data.cpu().numpy())
        imggt= imggt.transpose((1,2,0))
        
        
        
        #img=imgcom.resize((w, h),Image.ANTIALIAS)
        
        imggt= Image.fromarray(imggt)

        #img2=img2.resize((w, h),Image.ANTIALIAS)
        
        # calclate psnr and ssim
        # pdb.set_trace()
        #po=measure.compare_psnr(np.array(imggt),np.array(img),255)
        #p=measure.compare_psnr(np.array(imggt),np.array(img2),255)
        
        #so=measure.compare_ssim(np.array(imggt),np.array(img),multichannel=True,data_range=255)
        #s=measure.compare_ssim(np.array(imggt),np.array(img2),multichannel=True,data_range=255)
        
        img_luma = color.rgb2ycbcr(np.array(img)[:,:,:])[..., 0]
        imggt_luma = color.rgb2ycbcr(np.array(imggt)[:,:,:])[..., 0]
        img2_luma = color.rgb2ycbcr(np.array(img2)[:,:,:])[..., 0]

        po=cal_psnr(np.array(imggt_luma),np.array(img_luma))
        p=cal_psnr(np.array(imggt_luma),np.array(img2_luma))
        
        so=measure.compare_ssim(np.array(imggt_luma),np.array(img_luma),data_range=255)
        s=measure.compare_ssim(np.array(imggt_luma),np.array(img2_luma),data_range=255)

        psnr_ori=psnr_ori+po 
        psnr_pro=psnr_pro+p
        
        ssim_ori=ssim_ori+so 
        ssim_pro=ssim_pro+s
        
        
        print('psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f  mse:%f' % (po, p,so, s,mse))
        log.write('%d: psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f  mse:%f\n' % (i,po, p,so, s,mse))
        log.write('%s\n'%fn)
        if args.viz:
            h,w=img2.size
            imgout = np.zeros((w,3*h,3))
            imgout[0:w,0:0+h]=np.array(img)
            imgout[0:w,h:h+h]=np.array(img2)
            imgout[0:w,h*2:h*2+h]=np.array(imggt)
            imgout = Image.fromarray(imgout.astype(np.uint8))
            imgout.save('./testresult/output_%4d.jpg'%i)
            '''            
            #print('save img: %d' %i)
            h,w=img2.size
            imgout = np.zeros((h,3*w,3))
            imgout[0:256,0:0+w]=np.array(img)
            imgout[0:256,256:256+w]=np.array(img2)
            imgout[0:256,256*2:256*2+w]=np.array(imggt)
            imgout = Image.fromarray(imgout.astype(np.uint8))
            imgout.save('./testresult/output_%4d.jpg'%i)
            #img2.save('./testresult/out_%d.jpg' %i)
            #imggt.save('./testresult/gt_%d.jpg'%i)
            #img.save('./testresult/ori_%d.jpg'%i)
            '''
        
    psnr_ori=psnr_ori/len(im_files)
    psnr_pro=psnr_pro/len(im_files)
    ssim_ori=ssim_ori/len(im_files)
    ssim_pro=ssim_pro/len(im_files)
    print('psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f' % (psnr_ori, psnr_pro,ssim_ori, ssim_pro))
    log.close()
        

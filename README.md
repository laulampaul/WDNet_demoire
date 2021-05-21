This is the implementation of the paper Wavelet-based and Dual-branch Neural Network for Demoireing

# WDNet
Wavelet-based and Dual-branch Neural Network for Demoireing ECCV 2020

[\[Paper Link\]](https://arxiv.org/abs/2007.07173) 

## Abstract
When smartphone cameras are used to take photos of digital screens, usually moire patterns result, severely degrading photo quality. In this paper, we design a wavelet-based dual-branch network (WDNet) with a spatial attention mechanism for image demoireing. Existing image restoration methods working in the RGB domain have difficulty in distinguishing moire patterns from true scene texture. Unlike these methods, our network removes moire patterns in the wavelet domain to separate the frequencies of moire patterns from the image content. The network combines dense convolution modules and dilated convolution modules supporting large receptive fields. Extensive experiments demonstrate the effectiveness of our method, and we further show that WDNet generalizes to removing moire artifacts on non-screen images. Although designed for image demoireing, WDNet has been applied to two other low-levelvision tasks, outperforming state-of-the-art image deraining and derain-drop methods on the Rain100h and Raindrop800 data sets, respectively.

## start training
1. Download the dataset from [\[TIP2018 Dataset\]](https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC?usp=sharing)

2. Donwload the wavelet parameters from [\[wavelet parameters\]](https://github.com/hhb072/WaveletSRNet）

3. Put it into /cache/TrainData

4. Run the training_wdnet.py

## Citation
If you use this code, please cite:
@InProceedings{liu2020waveletbased,
      title={Wavelet-Based Dual-Branch Network for Image Demoireing}, 
      author={Lin Liu and Jianzhuang Liu and Shanxin Yuan and Gregory Slabaugh and Ales Leonardis and Wengang Zhou and Qi Tian},
      booktitle={ECCV, 2020}
}

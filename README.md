# Remote-sensing-image-semantic-segmentation  
The project uses Unet-based improved networks to study Remote sensing image semantic segmentation, which is based on keras.  
This project has been used in the Sparse Representation and Intelligent Analysis of 2019 Remote Sensing Image competition.  
（这工程已经在2019年遥感图像稀疏表征与智能分析竞赛中被使用过。）  

![](https://github.com/TachibanaYoshino/Remote-sensing-image-semantic-segmentation/blob/master/illustration.png)  

----  
## Requirements  
- python 3.6.8  
- tensorflow-gpu 1.8 
- Keras 2.2.4
- opencv-python  
- tqdm  
- numpy  
- glob  
- argparse  
- matplotlib  
- tifffile  
- pyjson  
- Pillow 6.0  
- scikit-learn  

## Usage  
### 1. Download dataset  
> Link  

### 2. Create new labels  
  `python create_train_val_label.py`  

### 3. Train  
  eg. `python train6_5.py --model checkpoint6_5`  
  
### 4. Download pre-trained weights  
> [Link](https://github.com/TachibanaYoshino/Remote-sensing-image-semantic-segmentation/releases/tag/checkpoint6_6)  

### 5. Test  
  eg. `python test.py --model 'checkpoint6_6'+ '/' + 'weights-039-0.7205-0.8099.h5'`  

## Results 
![]()  

from PIL import Image
import numpy as np
import os,json
from config import Config
from tqdm import tqdm
import argparse
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
from net.keras_layer_L2Normalization import L2Normalization
from net.group_norm import GroupNormalization
from Loss import lovasz_softmax
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

c = Config()
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
n_label = c.n_label
labelencoder = LabelEncoder()
labelencoder.fit(classes)

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",default='checkpoint'+ '/' + 'weights-003-0.6879-0.9207.h5',
        help="path to trained model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=c.size_train[0])
    ap.add_argument("-p", "--test_path", required=False,
        help="test file", type=str, default=c.test_path)
    args = vars(ap.parse_args())    
    return args

def out_mask_to_color_pic(mask, palette_file='Palette.json'):
    assert len(mask.shape) == 2
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
    color_pic = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in tqdm(range(0, mask.shape[0])):
        for j in range(0, mask.shape[1]):
            assert str(mask[i,j]) in list(text.keys())
            color_pic[i,j,:] = text[str(mask[i,j])]
    return color_pic
    
def predict(args):
    img_path = []
    for pic in os.listdir(args["test_path"]):
        if ' (2)' in pic:
            print(pic)
            img_path.append(args["test_path"] + '/' + pic)
    # load the trained convolutional neural network
    print("[INFO] loading network...")

    co = {'L2Normalization': L2Normalization, 'lovasz_softmax':lovasz_softmax,'tf':tf}
    model = load_model(args["model"], custom_objects= co)
    stride = args['stride']//2
    for n in range(len(img_path)):
        print(n)
        path = img_path[n]
        print(path.split('/')[-1][:-8])
        #load the image
        image = Image.open(path).convert("CMYK")
        image = np.asarray(image)
        h,w,_ = image.shape
        padding_h = (h//stride + 1 ) * stride
        padding_w = (w//stride + 1 ) * stride
        padding_img = np.zeros((padding_h,padding_w,4),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0

        print ('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+c.size_train[0], j*stride: j*stride+c.size_train[1], :]
                ch,cw,_ = crop.shape

                if ch != c.size_train[0] or cw != c.size_train[1]:
                    print ('invalid size!')
                    print(crop.shape)
                    continue
                    
                crop = np.expand_dims(crop, axis=0)
                #print 'crop:',crop.shape
                # pred = model.predict_classes(crop,verbose=2)
                pred = model.predict(crop,verbose=2)
                pred = pred[0]
                pred = np.reshape(pred, (1, c.size_train[1] * c.size_train[0], c.n_label))
                pred = np.argmax(pred, axis=2)
                # print(pred.shape)

                pred = labelencoder.inverse_transform(pred[0])  
                #print (np.unique(pred))  
                pred = pred.reshape((c.size_train[0],c.size_train[1])).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+c.size_train[0],j*stride:j*stride+c.size_train[1]] = pred[:,:]


        out = out_mask_to_color_pic(mask_whole[0:h,0:w])
        c.check_folder('./predict/')
        Image.fromarray(out).save('./predict/'+path.split('/')[-1][:-8]+'_label.tif', dpi = c.dpi)
        
    

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)




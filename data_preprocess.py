import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm
from config import Config
c = Config()
np.random.seed(c.seed)

# crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def random_crop(img1, img2, crop_H, crop_W):

    assert  img1.size[:2] ==  img2.size[:2]
    w, h = img1.size[:2]

    # 裁剪宽度不可超过原图可裁剪宽度
    if crop_W > w:
        crop_W = w
    # 裁剪高度

    if crop_H > h:
        crop_H = h

    # 随机生成左上角的位置
    x0 = random.randrange(0, w - crop_W + 1, 50)
    y0 = random.randrange(0, h - crop_H + 1, 50)

    crop_1 = img1.crop((x0, y0, x0+crop_W, y0+crop_H))
    crop_2 = img2.crop((x0, y0, x0+crop_W, y0+crop_H))

    return crop_1,crop_2

def rotate(x, y, angle):
    if angle ==90:
        x = x.transpose(Image.ROTATE_90)
        y = y.transpose(Image.ROTATE_90)
    elif angle == 180:
        x = x.transpose(Image.ROTATE_180)
        y = y.transpose(Image.ROTATE_180)
    elif angle == 270:
        x = x.transpose(Image.ROTATE_270)
        y = y.transpose(Image.ROTATE_270)
    return x, y

def flip(x,y):
    if np.random.randint(0, 2) == 0:
        x = x.transpose(Image.FLIP_LEFT_RIGHT)
        y = y.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.randint(0, 2) == 0:
        x = x.transpose(Image.FLIP_TOP_BOTTOM)
        y = y.transpose(Image.FLIP_TOP_BOTTOM)
    return x,y


def data_augment(x, y):

    if np.random.randint(0, 2) == 0:
        x, y = rotate(x, y, random.choice([90,180,270]))

    if np.random.randint(0, 2) == 0:
        x, y = flip(x,y)

    return x, y

def save(img,label,s, count, file):
    Img = Image.fromarray(img).convert("CMYK")
    Label = Image.fromarray(label)
    assert img.shape[:2] == label.shape[:2]
    if img.shape[0] != c.size_train[0] or img.shape[1] != c.size_train[1]:
        # resize    ANTIALIAS  and BILINEAR(no ANTIALIAS)
        Img, Label = Img.resize((c.size_train[1], c.size_train[0]), Image.ANTIALIAS), Label.resize(
            (c.size_train[1], c.size_train[0]), Image.BILINEAR)

    Img.save(file +'/img_'+ str(s)  +'_'+ str(count)  + '.tif', dpi=c.dpi)
    Label.save(file +'/label_'+ str(s)  +'_'+str(count)  + '.tif', dpi=(1.0,1.0))


def creat_train_dataset1(train_path, crop_size,prepro_train_path):
    print('creating dataset...')
    img_path = []
    label_path = []
    for pic in os.listdir(train_path):
        if 'new-L' in pic:
            label_path.append(train_path + '/' + pic)
            img_path.append(train_path + '/' + pic[:-10] + " (2).tif")
    assert len(img_path) == len(label_path)


    for s in tqdm(range(len(img_path))):

        src_img = Image.open(img_path[s]).convert("CMYK")
        label_img = Image.open(label_path[s])  # single channel
        img = np.asarray(src_img)
        label = np.asarray(label_img)
        h, w, ch = img.shape
        assert ch == 4
        count = 0
        for random_size in range(crop_size[0], crop_size[1]+1, 100):
            #  random_crop size
            stride = random_size

            for i in range(h // stride):
                for j in range(w // stride):
                    count += 1
                    crop_img = img[i * stride:i * stride + random_size, j * stride: j * stride + random_size, :]
                    crop_label = label[i * stride:i * stride + random_size, j * stride: j * stride + random_size]

                    save(crop_img,crop_label,s,count,prepro_train_path)

            if h % stride != 0:
                for i in range(w // stride):
                    count += 1
                    res = h % stride
                    crop_img = img[(h - res):, i * stride: i * stride + random_size, :]
                    crop_label = label[(h - res):, i * stride: i * stride + random_size]
                    save(crop_img, crop_label, s, count, prepro_train_path)

            if w % stride != 0:
                for i in range(h // stride):
                    count += 1
                    res = w % stride
                    crop_img = img[i * stride:i * stride + random_size, (w - res):, :]
                    crop_label = label[i * stride:i * stride + random_size, (w - res):]
                    save(crop_img, crop_label, s, count, prepro_train_path)

            if h % stride != 0 and w % stride != 0:
                res_h = h % stride
                res_w = w % stride
                crop_img = img[(h - res_h):, (w - res_w):, :]
                crop_label = label[(h - res_h):, (w - res_w):]
                count += 1
                save(crop_img, crop_label, s, count, prepro_train_path)

def creat_train_dataset(train_path, crop_size,prepro_train_path):
    print('creating dataset...')
    img_path = []
    label_path = []
    for pic in os.listdir(train_path):
        if 'new-L' in pic:
            label_path.append(train_path + '/' + pic)
            img_path.append(train_path + '/' + pic[:-10] + " (2).tif")
    assert len(img_path) == len(label_path)


    for s in tqdm(range(len(img_path))):

        src_img = Image.open(img_path[s]).convert("CMYK")
        label_img = Image.open(label_path[s])  # single channel
        img = np.asarray(src_img)
        label = np.asarray(label_img)
        h, w, ch = img.shape
        assert ch == 4
        count = 0

        for random_size in range(crop_size[0], crop_size[1]+1, 400):
            #  random_crop size
            stride = random_size // 2
            for i in range(h // stride - (random_size//stride-1)):
                for j in range(w // stride -(random_size//stride-1)):
                    count += 1
                    crop_img = img[i * stride:i * stride + random_size, j * stride: j * stride + random_size, :]
                    crop_label = label[i * stride:i * stride + random_size, j * stride: j * stride + random_size]

                    save(crop_img,crop_label,s,count,prepro_train_path)



# ------------Check the preprocessed image

if __name__ == '__main__':
    train_path = c.train_path
    crop_size =  [400, 800] # [200, 800]
    file = c.train_data_path
    c.check_folder(file)
    creat_train_dataset(train_path, crop_size, file)
import numpy as np
from PIL import Image
import os,random
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from config import Config
c = Config()
from data_preprocess import *

np.random.seed(c.seed)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
n_label = c.n_label
labelencoder = LabelEncoder()
labelencoder.fit(classes)



def TrainImggenerator(train_data_path, img_size, batch_size, augment):
    img_list, label_list = [], []

    for pic in os.listdir(train_data_path):
        if 'new-L' in pic:
            label_list.append(Image.open(train_data_path + '/' + pic))
            img_list.append(Image.open(train_data_path + '/' + pic.replace("_new-L", " (2)")).convert("CMYK"))

    assert len(label_list) == len(img_list)

    batch = 0
    x_img_batch, y_label_batch, y_label_batch1, y_label_batch2, y_label_batch3, y_label_batch4 = [], [], [], [], [], []
    while True:

        #  random image
        for i in np.random.permutation(np.arange(len(img_list))):

            src_img = img_list[i]
            label_img = label_list[i]  # single channel


            random_size = random.randrange(400,800+1,400)
            img, label = random_crop(src_img,label_img,random_size,random_size)

            if img.size[1] != c.size_train[0] or img.size[0] != c.size_train[1]:
                # resize    ANTIALIAS  and BILINEAR(no ANTIALIAS)
                img, label = img.resize((c.size_train[1], c.size_train[0]), Image.ANTIALIAS), label.resize(
                    (c.size_train[1], c.size_train[0]), Image.BILINEAR)

            if augment:
                img, label = data_augment(img, label)

            Img = np.asarray(img).astype("float") / 255.0


            label1 = np.asarray(label)
            label2 = np.asarray(label.resize((img_size[1] // 2, img_size[0] // 2), Image.BILINEAR))
            label3 = np.asarray(label.resize((img_size[1] // 4, img_size[0] // 4), Image.BILINEAR))
            label4 = np.asarray(label.resize((img_size[1] // 8, img_size[0] // 8), Image.BILINEAR))

            x_img_batch.append(Img)


            y_label_batch1.append(label1)
            y_label_batch2.append(label2)
            y_label_batch3.append(label3)
            y_label_batch4.append(label4)

            batch += 1
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(x_img_batch)


                train_label1 = np.array(y_label_batch1).flatten()
                train_label1 = labelencoder.transform(train_label1)
                train_label1 = to_categorical(train_label1, num_classes=n_label)
                train_label1 = train_label1.reshape((batch_size, img_size[0] * img_size[1], n_label))

                train_label2 = np.array(y_label_batch2).flatten()
                train_label2 = labelencoder.transform(train_label2)
                train_label2 = to_categorical(train_label2, num_classes=n_label)
                train_label2 = train_label2.reshape((batch_size, img_size[0] * img_size[1] // 4, n_label))

                train_label3 = np.array(y_label_batch3).flatten()
                train_label3 = labelencoder.transform(train_label3)
                train_label3 = to_categorical(train_label3, num_classes=n_label)
                train_label3 = train_label3.reshape((batch_size, img_size[0] * img_size[1] // 16, n_label))

                train_label4 = np.array(y_label_batch4).flatten()
                train_label4 = labelencoder.transform(train_label4)
                train_label4 = to_categorical(train_label4, num_classes=n_label)
                train_label4 = train_label4.reshape((batch_size, img_size[0] * img_size[1] // 64, n_label))

                yield (train_data, [train_label1, train_label2, train_label3, train_label4])
                x_img_batch, y_label_batch1, y_label_batch2, y_label_batch3, y_label_batch4, = [], [], [], [], []
                batch = 0








def ValImggenerator(data_path, img_size, batch_size):
    img_path = []
    label_path = []
    for pic in os.listdir(data_path):
        if 'new-L' in pic:
            label_path.append(data_path + '/' + pic)
            img_path.append(data_path + '/' + pic[:-10]+' (2).tif')
    assert len(img_path) == len(label_path)

    while True:
        batch = 0
        x_img_batch, y_label_batch1, y_label_batch2, y_label_batch3, y_label_batch4, = [], [], [], [], []
        for i in (range(len(img_path))):
            img = img_path[i]
            label = label_path[i]
            assert img[:-8] == label[:-10]
            img = Image.open(img).convert("CMYK")
            label = Image.open(label)
            img = np.asarray(img).astype("float") / 255.0
            label = np.asarray(label)
            assert img.shape[0:2] == label.shape[0:2]

            for i in range(img.shape[0]//img_size[0]):
                for j in range(img.shape[1]//img_size[1]):
                    x =   img[i * img_size[0] :(i + 1) * img_size[0],(j * img_size[1]) :(j + 1) * img_size[1],:]
                    y = label[i * img_size[0] :(i + 1) * img_size[0],(j * img_size[1]) :(j + 1) * img_size[1]]

                    x_img_batch.append(x)

                    label2 = np.asarray(Image.fromarray(y).resize((img_size[1] // 2, img_size[0] // 2), Image.BILINEAR))
                    label3 = np.asarray(Image.fromarray(y).resize((img_size[1] // 4, img_size[0] // 4), Image.BILINEAR))
                    label4 = np.asarray(Image.fromarray(y).resize((img_size[1] // 8, img_size[0] // 8), Image.BILINEAR))

                    y_label_batch1.append(y)
                    y_label_batch2.append(label2)
                    y_label_batch3.append(label3)
                    y_label_batch4.append(label4)

                    batch += 1
                    if batch % batch_size == 0:
                        # print 'get enough bacth!\n'
                        train_data = np.array(x_img_batch)


                        train_label1 = np.array(y_label_batch1).flatten()
                        train_label1 = labelencoder.transform(train_label1)
                        train_label1 = to_categorical(train_label1, num_classes=n_label)
                        train_label1 = train_label1.reshape((batch_size, img_size[0] * img_size[1], n_label))

                        train_label2 = np.array(y_label_batch2).flatten()
                        train_label2 = labelencoder.transform(train_label2)
                        train_label2 = to_categorical(train_label2, num_classes=n_label)
                        train_label2 = train_label2.reshape((batch_size, img_size[0] * img_size[1] // 4, n_label))

                        train_label3 = np.array(y_label_batch3).flatten()
                        train_label3 = labelencoder.transform(train_label3)
                        train_label3 = to_categorical(train_label3, num_classes=n_label)
                        train_label3 = train_label3.reshape((batch_size, img_size[0] * img_size[1] // 16, n_label))

                        train_label4 = np.array(y_label_batch4).flatten()
                        train_label4 = labelencoder.transform(train_label4)
                        train_label4 = to_categorical(train_label4, num_classes=n_label)
                        train_label4 = train_label4.reshape((batch_size, img_size[0] * img_size[1] // 64, n_label))

                        yield (train_data, [ train_label1, train_label2, train_label3, train_label4])
                        x_img_batch, y_label_batch1, y_label_batch2, y_label_batch3, y_label_batch4, = [], [], [], [], []
                        batch = 0


if __name__ == '__main__':
    val_path = c.val_path

    a = ValImggenerator(val_path, c.size_train,c.batch_size)
    print(next(a))


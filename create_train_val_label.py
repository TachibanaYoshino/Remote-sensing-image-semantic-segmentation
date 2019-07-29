import json,os
import numpy as np
import tifffile as tiff



def get_label_from_palette(label_img, palette_file='Palette.json'):
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
        label = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
        for i in range(label_img.shape[0]):
            print(i)
            for j in range(label_img.shape[1]):

                assert list(label_img[i, j, :]) in list(text.values())

                label[i, j] = int(list(text.keys())[list(text.values()).index(list(label_img[i, j, :]))])

        return label

def main(path):
    for pic in os.listdir(path):
        if 'label' in pic:
            print(pic)
            label = tiff.imread(path + '/' +pic)
            label = get_label_from_palette(label)
            tiff.imsave(path + '/' +pic[:-9] + 'new-L.tif',label)


if __name__ == '__main__':
    train_path = 'dataset/train'
    val_path = 'dataset/val'
    main(train_path)
    main(val_path)




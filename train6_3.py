# coding=utf-8

import argparse,time
import numpy as np
from net.model6_3 import Net
from config import Config
from data_loader5 import *
from keras.callbacks import ModelCheckpoint
import os
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import generic_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
c = Config()
c.batch_size = 2

def train(args):
    EPOCHS = c.num_epochs

    train_numb = 1777 * c.batch_size
    # val img patch number
    valid_numb = len([x for x in os.listdir(c.val_path) if '(2)' in x]) * 6800 // 400 * 7200 // 400

    input = Input(shape=(c.size_train[0], c.size_train[1], 4))
    pred = Net(c.n_label, input)
    model = Model(input, pred)

    model.summary()
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=adam, loss=['categorical_crossentropy' for _ in range(5)],
                  loss_weights=[10., 0.1, 0.1, 0.1, 0.1], metrics=['accuracy'])

    train_set = TrainImggenerator(c.train_path, c.size_train, c.batch_size, c.augment)
    val_set = ValImggenerator(c.val_path, c.size_train, c.batch_size)

    if os.path.exists(args['model']) and os.listdir(args['model']):
        a = sorted(file for file in os.listdir(args['model']))
        model.load_weights(args['model'] + '/' + a[-1], by_name=True)
        # 若成功加载前面保存的参数，输出下列信息
        print('-------------------' + args['model'] + '/' + a[-1] + "----checkpoint_loaded")

    iter_num = 0
    epoch_length = train_numb // c.batch_size
    for epoch_num in range(c.num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1 , c.num_epochs))
        epoch_start_time = time.time()
        while True:
            try:

                X, Y = next(train_set)
                H = model.train_on_batch(X, Y)
                # ---- -空间网络设置权重共享
                for l in model.layers:

                    if l.name[:2] == 'ud':
                        weitht_udconv = model.get_layer(name='udconv').get_weights()
                        l.set_weights(weitht_udconv)

                    if l.name[:2] == 'du':
                        weitht_duconv = model.get_layer(name='duconv').get_weights()
                        l.set_weights(weitht_duconv)

                iter_num += 1

                progbar.update(iter_num, [('loss',H[0]),('l0_loss',H[1]),('l1_loss',H[2]),('l2_loss',H[3]),('l3_loss',H[4]),('l4_loss',H[5]),
                                         ('l0_acc',H[6]),('l1_acc',H[7]),('l2_acc',H[8]),('l3_acc',H[9]),('l4_acc',H[10])])
                if iter_num == epoch_length:
                    hist = np.zeros((valid_numb//c.batch_size, 11))
                    for i in range(valid_numb//c.batch_size):
                        U, V = next(val_set)
                        H_val = model.test_on_batch(U,V)

                        for j in range(11):
                            hist[i][j] = H_val[j]
                    b = np.mean(hist,axis=0)
                    print(f'epoch time: {time.time()-epoch_start_time:.4f} s, vali--: loss:{b[0]:.4f},l0_loss:{b[1]:.4f},l1_loss:{b[2]:.4f},l2_loss:{b[3]:.4f},l3_loss:{b[4]:.4f},l4_loss:{b[5]:.4f}'+
                                         '\n'+f'                                            l0_acc:{b[6]:.4f},l1_acc:{b[7]:.4f},l2_acc:{b[8]:.4f},l3_acc:{b[9]:.4f},l4_acc:{b[10]:.4f}')
                    model.save(os.path.join(args['model'], 'weights_{:03d}_{:.4f}.h5'.format(epoch_num + 1, b[6])))
                    iter_num = 0
                    break
            except Exception as e:
                print ('Exception: {}'.format(e))
                break

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default='checkpoint6_3',
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    c.check_folder(args['model'])
    train(args)


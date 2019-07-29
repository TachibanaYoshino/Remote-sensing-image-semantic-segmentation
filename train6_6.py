# coding=utf-8
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from net.model6_6 import Net
from config import Config
from data_loader5 import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import os
from keras.layers import Input
from keras.models import Model
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
c = Config()
c.batch_size = 7

def train(args):
    EPOCHS = c.num_epochs
    filepath = "weights-{epoch:03d}-{val_l0_acc:.4f}-{val_l0_loss:.4f}.h5"
    model_weights = os.path.join(args['model'], filepath)

    train_numb = 1777 * c.batch_size
    # val img patch number
    valid_numb = len([x for x in os.listdir(c.val_path) if '(2)' in x]) * 6800 // 400 * 7200 // 400

    input = Input(shape=(c.size_train[0], c.size_train[1], 4))
    pred = Net(c.n_label, input)
    model = Model(input, pred)

    model.summary()
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=adam, loss=['categorical_crossentropy' for _ in range(5)],
                  loss_weights=[10., 0.1, 0.1, 0.1, 0.1], metrics=['accuracy'])

    modelcheck = ModelCheckpoint(model_weights, monitor='val_l0_acc', verbose=1, save_best_only=True, mode='auto')
    # reduceLROnPlat = ReduceLROnPlateau(monitor='val_l0_loss', factor=0.2, patience=5, verbose=1, mode='min',epsilon=0.0001)
    callable = [modelcheck]

    train_set = TrainImggenerator(c.train_path, c.size_train, c.batch_size, c.augment)
    val_set = ValImggenerator(c.val_path, c.size_train, c.batch_size)

    if os.path.exists(args['model']) and os.listdir(args['model']):
        a = sorted(file for file in os.listdir(args['model']))
        model.load_weights(args['model'] + '/' + a[-1], by_name=True)
        # 若成功加载前面保存的参数，输出下列信息
        print('-------------------' + args['model'] + '/' + a[-1] + "----checkpoint_loaded")

    H = model.fit_generator(generator=train_set, steps_per_epoch=train_numb // c.batch_size, epochs=c.num_epochs,
                            verbose=1,
                            validation_data=val_set, validation_steps=valid_numb // c.batch_size, callbacks=callable,
                            max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["l0_acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_l0_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default='checkpoint6_6',
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    c.check_folder(args['model'])
    train(args)


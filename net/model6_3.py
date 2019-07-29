#coding=utf-8

from keras import regularizers
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Lambda,Activation,Deconv2D,Concatenate,Add
from keras import backend as K
from .keras_layer_L2Normalization import L2Normalization
import tensorflow as tf

if K.image_dim_ordering() == 'tf':
    bn_axis = 3
else:
    bn_axis = 1




def conv_block(input_tensor, filters, kernel_size, name, strides, padding='same', dila=1):
    x = Conv2D(filters, kernel_size, strides=strides, name= name, padding=padding, kernel_initializer='he_uniform',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),dilation_rate=dila)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_'+ name)(x)
    x = Activation('relu')(x)
    return x

def split(x):
    return tf.split(x,x.shape[-1]//16,-1)

def udconv(x,name='udconv'):
    x = Activation('relu')(Conv2D(16, (3,3), strides=1, padding='same', name=name, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(x))
    return x

def duconv(x,name='duconv'):
    x = Activation('relu')(Conv2D(16, (3,3), strides=1, padding='same',name=name, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(x))
    return x
# -----

def Net(n_label,img_input):

    # ---------left branch -----
    x = conv_block(img_input, 32, (3, 3), strides=1, name='L_conv1-1')
    L1 = conv_block(x, 32, (3, 3), strides=1, name='L_conv1-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L1)
    #   400 -> 200

    x  = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-1')
    L2 = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L2)
    #   200 -> 100

    x  = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-1')
    L3 = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L3)
    #   100 -> 50

    x  = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-1')
    L4 = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L4)
    #   50 -> 25

    x = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-1')
    x = conv_block(x, 512, (3, 3), strides=1,dila=2, name='L_conv5-2')
    x = conv_block(x, 512, (3, 3), strides=1,dila=2, name='L_conv5-3')
    L5 = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-4')
    #    25

    # ---------Right branch -----

    #   25 -> 50
    x = Deconv2D(256, kernel_size=2, strides=2, padding='same',name='R_conv1-1')(L5)
    x = BatchNormalization(axis=bn_axis, name='R_conv1-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L4]), 256, (3, 3), strides=1, name='R_conv1-2')
    x = conv_block(x, 256, (3, 3), strides=1, name='R_conv1-3')
    R_out1 = Conv2D(n_label,(1,1),name='R_out1')(x)

    #   50 -> 100
    x = Deconv2D(128, kernel_size=2, strides=2, padding='same', name='R_conv2-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv2-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L3]), 128, (3, 3), strides=1, name='R_conv2-2')
    x = conv_block(x, 128, (3, 3), strides=1, name='R_conv2-3')
    R_out2 = Conv2D(n_label, (1, 1), name='R_out2')(x)

    #   100 -> 200
    x = Deconv2D(64, kernel_size=2, strides=2, padding='same', name='R_conv3-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv3-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L2]), 64, (3, 3), strides=1, name='R_conv3-2')
    x = conv_block(x, 64, (3, 3), strides=1, name='R_conv3-3')
    R_out3 = Conv2D(n_label, (1, 1), name='R_out3')(x)

    #   200 -> 400
    x = Deconv2D(32, kernel_size=2, strides=2, padding='same', name='R_conv4-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv4-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L1]), 32, (3, 3), strides=1, name='R_conv4-2')
    x = conv_block(x, 32, (3, 3), strides=1, name='R_conv4-3')
    R_out4 = Conv2D(n_label, (1, 1), name='R_out4')(x)

    # ---------Recoding branch -----

    x = conv_block(R_out4, 32, (3, 3), strides=1, name='E_conv1-1')
    x = conv_block(x, 32, (3, 3), strides=1, name='E_conv1-2')
    x = conv_block(x, 32, (3, 3), strides=2, name='E_conv1-3')
    #   400 -> 200

    x = conv_block(Concatenate(axis=-1)([x, R_out3]), 64, (3, 3), strides=1, name='E_conv2-1')
    p1 = conv_block(x, 64, (3, 3), strides=1, name='E_conv2-2')
    x = conv_block(p1, 64, (3, 3), strides=2, name='E_conv2-3')
    #   200 -> 100

    x = conv_block(Concatenate(axis=-1)([x, R_out2]), 128, (3, 3), strides=1, name='E_conv3-1')
    x = conv_block(x, 128, (3, 3), strides=1, name='E_conv3-2')
    x = conv_block(x, 128, (3, 3), strides=2, name='E_conv3-3')
    #   100 -> 50

    x = conv_block(Concatenate(axis=-1)([x, R_out1]), 256, (3, 3), strides=1, name='E_conv4-1')
    x = conv_block(x, 256, (3, 3), strides=1, name='E_conv4-2')
    x = conv_block(x, 256, (3, 3), strides=1, dila=2, name='E_conv4-3')
    x = conv_block(x, 256, (3, 3), strides=1, dila=2, name='E_conv4-4')
    x = conv_block(x, 256, (3, 3), strides=1, name='E_conv4-5')
    #   50

    # ----------空间网络----------------
    # up to down
    feature_list_new = []
    f= Lambda(split,  name='spilt')(x)
    feature_list_new.append(f[0])

    conv1 = Add()([udconv(f[0]), f[1]])
    feature_list_new.append(conv1)

    for cnt in range(2, len(f)):
        conv1 = Add()([udconv(feature_list_new[cnt - 1],name='ud'+str(cnt)), f[cnt]])
        feature_list_new.append(conv1)


    # down to top #
    f2 = feature_list_new
    feature_list_new = []
    length = len(f2) - 1
    feature_list_new.append(f2[length])
    conv2 = Add()([duconv(f2[length]), f2[length - 1] ])
    feature_list_new.append(conv2)

    for cnt in range(2, length+1):
        conv2 =Add()([duconv(feature_list_new[cnt - 1],name='du'+str(cnt)), f2[length - cnt] ])
        feature_list_new.append(conv2)

    feature_list_new.reverse()
    processed_feature = Concatenate(axis=-1)(feature_list_new)

# -----------------------------------------
    final_out = Conv2D(n_label,(1,1), name='final_out')(processed_feature)
    final_out = UpSampling2D(size=(8,8))(final_out)

    final_out = Activation('softmax',name='l0')(Reshape((400 * 400, n_label))(final_out))
    out1 = Activation('softmax',name='l1')(Reshape((400 * 400, n_label))(R_out4))
    out2 = Activation('softmax',name='l2')(Reshape((200 * 200, n_label))(R_out3))
    out3 = Activation('softmax',name='l3')(Reshape((100 * 100, n_label))(R_out2))
    out4 = Activation('softmax',name='l4')(Reshape((50 * 50, n_label))(R_out1))

    return [final_out, out1, out2, out3, out4]



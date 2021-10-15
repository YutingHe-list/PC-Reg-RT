# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, UpSampling3D, Conv2D, Conv3D, Activation, MaxPooling3D

from layers_3d import *
from dense_3D_spatial_transformer import SpatialTransformer


img_height = 128
img_width = 144
img_depth = 112
channels = 5
img_size = img_height * img_width *img_depth

batch_size = 1
pool_size = 50
ngf = 32# Number of filters in first layer of generator
ndf = 64# Number of filters in first layer of discriminator

def unet(input, name="unet", train=True):
    with tf.variable_scope(name):
        y1 = UConv(input, 8, strides=(1, 1, 1), is_training=train, name='d1')
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(y1)

        y2 = UConv(pool1, 16, strides=(1, 1, 1), is_training=train, name='d2')
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(y2)

        y3 = UConv(pool2, 32, strides=(1, 1, 1), is_training=train, name='d3')
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(y3)

        y4 = UConv(pool3, 64, strides=(1, 1, 1), is_training=train, name='d4')

        up3 = UpSampling3D(size=(2, 2, 2))(y4)
        merge3 = tf.concat([y3, up3], -1)
        db3 = UConv(merge3, 32, strides=(1, 1, 1), is_training=train, name='d5')
        su3 = Conv3D(5, 1, kernel_initializer='he_normal', padding='same')(db3)
        su3 = Conv3D(5, 3, kernel_initializer='he_normal', padding='same')(UpSampling3D(size=(2, 2, 2))(su3))

        up2 = UpSampling3D(size=(2, 2, 2))(db3)
        merge2 = tf.concat([y2, up2], -1)
        db2 = UConv(merge2, 16, strides=(1, 1, 1), is_training=train, name='d6')
        su2 = Conv3D(5, 1, kernel_initializer='he_normal', padding='same')(db2)
        su2 = su3 + su2
        su2 = Conv3D(5, 3, kernel_initializer='he_normal', padding='same')(UpSampling3D(size=(2, 2, 2))(su2))

        up1 = UpSampling3D(size=(2, 2, 2))(db2)
        merge1 = tf.concat([y1, up1], -1)
        db1 = UConv(merge1, 8, strides=(1, 1, 1), is_training=train, name='d7')
        output = Conv3D(5, 1, kernel_initializer='he_normal', padding='same')(db1)
        output = output + su2
        output = Activation('softmax')(output)
        return output

def build_generator_unet(src, label, src_org, tgt, enc_nf, dec_nf, stride_z=2, full_size=True, train=True, name="generator"):
    with tf.variable_scope(name):
        x_in = tf.concat([src[:,:,:,:,:], tgt[:,:,:,:,:]], -1)

        # down-sample path.
        x0 = myConv(x_in, enc_nf[0], strides=(2, 2, 2), is_training=train, name='c1')  # 80x96x112
        x1 = myConv(x0, enc_nf[1], (2, 2, stride_z), train, "c2")  # 40x48x56
        x2 = myConv(x1, enc_nf[2], (2, 2, stride_z), train, "c3")  # 20x24x28
        x3 = myConv(x2, enc_nf[3], (2, 2, stride_z), train, "c4")  # 10x12x14

        # up-sample path.
        x = myConv(x3, dec_nf[0], 1, train, "c5")
        x = UpSampling3D(size=(2, 2, stride_z))(x)
        x = tf.concat([x, x2], -1)
        x = myConv(x, dec_nf[1], 1, train, "c6")
        x = UpSampling3D(size=(2, 2, stride_z))(x)
        # fit size
        x = tf.concat([x, x1],-1)
        x = myConv(x, dec_nf[2], 1, train, "c7")
        x = UpSampling3D(size=(2, 2, stride_z))(x)
        # fit size
        x = tf.concat([x, x0], -1)
        x = myConv(x, dec_nf[3], 1, train, "c8")
        x = myConv(x, dec_nf[4], 1, train, "c9")

        if full_size:
            x = UpSampling3D(size=(2, 2, 2))(x)
            x = tf.concat([x, x_in], -1)
            x = myConv(x, dec_nf[5], 1, train, "c10")

            # optional convolution
            if (len(dec_nf) == 8):
                x = myConv(x, dec_nf[6], 1, train, "c11")

        # transform the results into a flow.

        flow = flowconv(x, dec_nf[-1], name='flow')

        y = SpatialTransformer(interp_method='linear')([src, flow])
        y_org = SpatialTransformer(interp_method='linear')([src_org, flow])
        warp_label = SpatialTransformer(interp_method='nearest')([label, flow])

        return y, flow, y_org, warp_label







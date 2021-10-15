import tensorflow as tf
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.layers import Conv3D
from tensorflow.contrib.layers import batch_norm
# tf.python.keras


def myConv(x_in, nf, strides=1, is_training=True, name='conv3d'):
    with tf.variable_scope(name):
        # x_out = Conv3D(nf, kernel_size=3, padding='same',
        #        kernel_initializer='he_normal', strides=strides)(x_in)
        x_out = tf.layers.conv3d(inputs=x_in,
                                 filters=nf,
                                 kernel_size=(3, 3, 3),
                                 strides=strides,
                                 padding='same',
                                 kernel_initializer=he_normal(),
                                 )
        # x_out = batch_norm(x_out, is_training, name='bn')
        x_out = GroupNorm(x_out, is_training, name='bn', G=x_out.shape[-1]//4)
        x_out = LeakyRelU(x_out, 0.2)
        return x_out

def UConv(x_in, nf, strides=1, is_training=True, name='UConv'):
    with tf.variable_scope(name):
        x_out = tf.layers.conv3d(inputs=x_in,
                                 filters=nf,
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding='same',
                                 kernel_initializer=he_normal(),
                                 )
        # x_out = batch_norm(x_out, is_training, name='bn1')
        x_out = GroupNorm(x_out, is_training, name='bn1', G=x_out.shape[-1]//4)
        x_out = LeakyRelU(x_out, 0.2)

        x_out = tf.layers.conv3d(inputs=x_out,
                                 filters=nf,
                                 kernel_size=(3, 3, 3),
                                 strides=strides,
                                 padding='same',
                                 kernel_initializer=he_normal(),
                                 )
        # x_out = batch_norm(x_out, is_training, name='bn2')
        x_out = GroupNorm(x_out, is_training, name='bn2', G=x_out.shape[-1]//4)
        x_out = LeakyRelU(x_out, 0.2)
        return x_out

def flowconv(x,nf,name="flow"):
    with tf.variable_scope(name):
        return tf.layers.conv3d(x,
                                nf,
                                kernel_size=(3, 3, 3),
                                padding='same',
                                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5))

def LeakyRelU(x, leak=0.2, name="LeakyRelu", alt_relu_impl=True):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * tf.abs(x)
        else:
            return tf.maximum(x, leak*x)

# not sure to use
def myDeconv(x_in, nf, strides=1, name = 'deconv3d',do_norm=True, do_relu=True):
    with tf.variable_scope(name):
        initializer = tf.keras.initializers.he_normal()
        resized_input = tf.keras.layers.UpSampling3D()(x_in)
        x_out = tf.layers.conv3d_transpose(x_in, nf, kernel_size=3, padding='same', kernel_initializer=initializer, strides=strides)
    return x_out


def batch_norm(x, is_train=True, name='batch_norm'):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x,
                                             epsilon=1e-5,
                                             momentum=0.99,
                                             training=is_train)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out

def GroupNorm(x, is_training=True, G=8, eps=1e-5, name='group_norm'):
    with tf.variable_scope(name):
        N, H, W, D, C = x.shape
        # A=tf.cast(C // G, tf.int32)
        x = tf.reshape(x, [tf.cast(N, tf.int32), tf.cast(H, tf.int32), tf.cast(W, tf.int32), tf.cast(D, tf.int32),
                           tf.cast(G, tf.int32), tf.cast(C // G, tf.int32)])
        mean, var = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [tf.cast(N, tf.int32), tf.cast(H, tf.int32), tf.cast(W, tf.int32), tf.cast(D, tf.int32),
                           tf.cast(C, tf.int32)])
        gamma = tf.get_variable('gamma', shape=[1, 1, 1, 1, C],
                                initializer=tf.constant_initializer(1.))
        beta = tf.get_variable('beta', shape=[1, 1, 1, 1, C],
                                initializer=tf.constant_initializer(0.))
        # gamma = tf.Variable(tf.ones(shape=[1, 1, 1, 1, tf.cast(C, tf.int32)]), name=gamma_name)
        # beta = tf.Variable(tf.zeros(shape=[1, 1, 1, 1, tf.cast(C, tf.int32)]), name=beta_name)
        return x * gamma + beta

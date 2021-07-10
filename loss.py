
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.keras._impl.keras import backend as K

def crossEntropyLoss(y_true, y_pred):
    smooth = tf.constant(1e-6)
    crossentry = -tf.reduce_mean(y_true * tf.log(y_pred+smooth)+(1-y_true)*tf.log(1-y_pred+smooth))
    return crossentry
def crossEntropyLoss_multi(y_true, y_pred):
    smooth = tf.constant(1e-6)
    crossentry = -tf.reduce_mean(y_true * tf.log(y_pred+smooth))
    return crossentry

def inconsistencyLoss(y_fixed, y_mov):
    bs = tf.cast(tf.shape(y_fixed)[0], tf.float32)
    width = tf.cast(tf.shape(y_fixed)[1], tf.float32)
    height = tf.cast(tf.shape(y_fixed)[2], tf.float32)
    depth = tf.cast(tf.shape(y_fixed)[3], tf.float32)
    channel = tf.cast(tf.shape(y_fixed)[4], tf.float32)
    n = width * height * depth
    inc_loss = tf.reduce_sum(tf.abs(y_mov - y_fixed) ** 2) / (n * bs * channel)
    return inc_loss
def mean_squared_error(y_true, y_pred):
  return math_ops.reduce_mean(K.mean(math_ops.square(y_pred - y_true), axis=-1))
def mse(output, target, is_mean=True, name="mean_squared_error"):
    """Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : Tensor
        2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, height, width] or [batch_size, height, width, channel].
    target : Tensor
        The target distribution, format the same with `output`.
    is_mean : boolean
        Whether compute the mean or sum for each example.
            - If True, use ``tf.reduce_mean`` to compute the loss between one target and predict data.
            - If False, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`__

    """
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
        else:
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3, 4]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3, 4]))
        return mse
def tf_repeat(y, repeat_num):
    return tf.tile(tf.expand_dims(y, axis=-1), [1, 1, 1, 1, repeat_num])
    # return tf.reshape(tf.tile(tf.expand_dims(y, axis=-1), [1, 1, 1, 1, repeat_num]), [1, 12, 12, 2, 64])
def pearson(y_true, y_pred):
    mean1 = tf.reduce_mean(y_true, axis=-1)
    mean2 = tf.reduce_mean(y_pred, axis=-1)
    a = y_true.get_shape()[4]
    mean1_tile = tf_repeat(mean1, a)
    mean2_tile = tf_repeat(mean2, a)
    sum1sq = tf.reduce_sum(tf.square((y_true-mean1_tile)))
    sum2sq = tf.reduce_sum(tf.square((y_pred-mean2_tile)))
    top = tf.reduce_sum(tf.multiply((y_true-mean1_tile),(y_pred-mean2_tile)))
    den = tf.sqrt(tf.multiply(sum1sq, sum2sq))
    return tf.abs(top/den)

def KLDloss(y_true, y_pred):
    smooth = tf.constant(1e-6)
    loss = tf.reduce_sum(y_true*tf.log(y_true/(y_pred+smooth)+smooth))
    return loss

def diceLoss(y_true, y_pred):
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top / bottom)
    return (1-dice)

def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims+1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice

def gradientLoss(y_pred, penalty='l1'):

    dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
    dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
    return d/3.0

def cc3D(I, J, win=[9, 9, 9], voxel_weights=None):
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    I2 = I*I
    J2 = J*J
    IJ = I*J

    filt = tf.ones([win[0], win[1], win[2], 1, 1])

    I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
    J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
    I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
    J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
    IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

    win_size = win[0]*win[1]*win[2]
    u_I = I_sum/win_size
    u_J = J_sum/win_size

    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    cc = cross*cross / (I_var*J_var+1e-6)
    return -1.0*tf.reduce_mean(cc)

# cross-correlation
def cc4D(I, J, win=[9, 9, 9]):
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # compute CC squares
    I2 = I*I
    J2 = J*J
    IJ = I*J

    filt = tf.ones([win[0], win[1], win[2], 57, 1])

    I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
    J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
    I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
    J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
    IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

    win_size = win[0]*win[1]*win[2]
    u_I = I_sum/win_size
    u_J = J_sum/win_size

    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    cc = cross*cross / (I_var*J_var+1e-6)
    return -1.0*tf.reduce_mean(cc)

def lp_loss(vol_generated, vol_gt, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.
    @param ct_generated: The predicted ct
    @param gt_ct: The ground truth ct
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @return: The lp loss.
    """
    bs = tf.cast(tf.shape(vol_gt)[0], tf.float32)
    width = tf.cast(tf.shape(vol_gt)[1], tf.float32)
    height = tf.cast(tf.shape(vol_gt)[2], tf.float32)
    depth = tf.cast(tf.shape(vol_gt)[3], tf.float32)
    n = width*height*depth
    lp_loss=tf.reduce_sum(tf.abs(vol_generated - vol_gt)**l_num)/(n*bs)
    tf.add_to_collection('losses', lp_loss)

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

# image gradient loss
def gdl_loss(gen_frames, gt_frames, alpha):

    # calculate the loss for each scale
    scale_losses = []
    for i in range(len(gen_frames)):
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.pack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.pack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.pack(scale_losses))


def _validate_information_penalty_inputs(structured_generator_inputs,
                                         predicted_distributions):
  """Validate input to `mutual_information_penalty`."""
  _validate_distributions(predicted_distributions)
  if len(structured_generator_inputs) != len(predicted_distributions):
    raise ValueError(
        '`structured_generator_inputs` length %i must be the same '
        'as `predicted_distributions` length %i.' %
        (len(structured_generator_inputs), len(predicted_distributions)))

def _validate_distributions(distributions):
  if not isinstance(distributions, (list, tuple)):
    raise ValueError('`distributions` must be a list or tuple. Instead, '
                     'found %s.' % type(distributions))
  for x in distributions:
    # We used to check with `isinstance(x, tf.compat.v1.distributions.Distribution)`.
    # However, distributions have migrated to `tfp.distributions.Distribution`,
    # which is a new code repo, so we can't check this way anymore until
    # TF-GAN is migrated to a new repo as well.
    # This new check is not sufficient, but is a useful heuristic for now.
    if not callable(getattr(x, 'log_prob', None)):
      raise ValueError('`distributions` must be a list of `Distributions`. '
                       'Instead, found %s.' % type(x))

def mutual_information_penalty(structured_generator_inputs, predicted_distributions, scope=None):
    with ops.name_scope(scope, 'mutual_information_loss') as scope:
     total_losses = 0
     for i in range(1):
         total_losses =total_losses -(tf.reduce_mean(structured_generator_inputs[i,:,:,:,:]) - tf.log(tf.reduce_mean(tf.exp(predicted_distributions[i,:,:,:,:]))))
     return total_losses


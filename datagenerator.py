import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import transform
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform
def limit(image):
    max = np.where(image < 0)
    image[max] = 0
    return image

def standardization(image):
    image = image.astype('float32')
    a = image.mean()
    b = image.std()
    image = (image-a)/b
    return image

def repeat(img):
    shape = (96, 96, 12)
    img_new = np.zeros(shape)
    img_new[:, :, 0] = img[:, :, 0]
    img_new[:, :, 1:11] = img
    img_new[:, :, 11] = img[:, :, 9]
    return img_new

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == vals[c]] = vals[c]
    return res


def Nor(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

ids = 0

def example_gen_3D_brain(vol_names, bs, data_aug=True):
    #idx = 0
    global ids
    while(True):
        # print(len(vol_names))
        # idxes = np.random.randint(len(vol_names), size=bs)
        X_data = []
        Y_data = []
        R_data = []
        S_data = []
        if ids >= len(vol_names)//bs:
            ids = 0
        for idx in range(bs*ids, bs*(ids+1)):
            mov_img = sio.loadmat(vol_names[idx])['mov_img']
            fix_img = sio.loadmat(vol_names[idx])['fixed_img']
            mov_label = sio.loadmat(vol_names[idx])['mov_label']
            fix_label = sio.loadmat(vol_names[idx])['fixed_label']
            rnd_val = np.random.uniform()
            if rnd_val <= 0.5:
                mov_img, mov_label, fix_img, fix_label = SpatialTransform((128, 144, 112),
                                                                          list(np.array((128, 144, 112)) // 2),
                                                                          True, (100., 350.), (14., 17.),
                                                                          True, (0, 2. * np.pi), (-0.000001, 0.00001),
                                                                          (-0.000001, 0.00001),
                                                                          True, (0.7, 1.3), 'constant', 0, 3,
                                                                          'constant',
                                                                          0, 0,
                                                                          random_crop=False)(mov_img, mov_label,
                                                                                             fix_img, fix_label)

            mov_img = limit(mov_img)
            mov_img = Nor(mov_img)
            mov_img = np.reshape(mov_img, (1,) + mov_img.shape + (1,))

            fix_img = limit(fix_img)
            fix_img = Nor(fix_img)
            fix_img = np.reshape(fix_img, (1,) + fix_img.shape + (1,))

            fix_label_3 = fix_label.astype('int16')
            d = np.where(fix_label_3 > 4)
            fix_label_3[d] = 0
            fix_label_3 = tf.keras.utils.to_categorical(fix_label_3, num_classes=5)
            fix_label_3 = fix_label_3[np.newaxis, :, :, :, :]

            mov_label_3 = mov_label.astype('int16')
            d = np.where(mov_label_3 > 4)
            mov_label_3[d] = 0
            mov_label_3 = tf.keras.utils.to_categorical(mov_label_3, num_classes=5)
            mov_label_3 = mov_label_3[np.newaxis, :, :, :, :]


            X_data += [mov_img]
            Y_data += [fix_img]
            R_data += [fix_label_3]
            S_data += [mov_label_3]
        if bs > 1:
            return_vals = [np.concatenate(X_data, 0)]
            return_vals.append(np.concatenate(Y_data, 0))
            return_vals.append(np.concatenate(R_data, 0))
            return_vals.append(np.concatenate(S_data, 0))

        else:
            return_vals = [X_data[0]]
            return_vals.append(Y_data[0])
            return_vals.append(R_data[0])
            return_vals.append(S_data[0])
        ids += 1
        print('第%d次取数据'%(ids))

        yield tuple(return_vals)

def example_gen_3D_brain_ag(vol_names, bs, data_aug=True):
    #idx = 0
    global ids
    while(True):
        # print(len(vol_names))
        # idxes = np.random.randint(len(vol_names), size=bs)
        X_data = []
        Y_data = []
        R_data = []
        S_data = []
        F_data = []
        O_data = []
        if ids >= len(vol_names)//bs:
            ids = 0
        for idx in range(bs*ids, bs*(ids+1)):
            a = int(vol_names[idx].split('\\')[-1].split('_')[0])
            b = int(vol_names[idx].split('\\')[-1].split('_')[1].split('.')[0])
            if a <= 5 and b<=5:
                flag = 1
            if a<= 5 and b>5:
                flag = 2
            if a>5 and b<=5:
                flag = 3
            if a>5 and b>5:
                flag = 4
            mov_img = sio.loadmat(vol_names[idx])['mov_img']
            # mov_img = tf.image.resize_images(mov_img,(128,64,128),method=0)
            fix_img = sio.loadmat(vol_names[idx])['fixed_img']
            # fix_img = tf.image.resize_images(fix_img, (128, 64, 128), method=0)
            mov_label = sio.loadmat(vol_names[idx])['mov_label']
            # mov_label = tf.image.resize_images(mov_label, (128, 64, 128), method=1)
            fix_label = sio.loadmat(vol_names[idx])['fixed_label']
            # fix_label = tf.image.resize_images(fix_label, (128, 64, 128), method=1)
            rnd_val = np.random.uniform()
            if rnd_val <= 0.5:
                mov_img, mov_label, fix_img, fix_label = SpatialTransform((128, 144, 112),
                                                                          list(np.array((128, 144, 112)) // 2),
                                                                          True, (100., 350.), (14., 17.),
                                                                          True, (0, 2. * np.pi), (-0.000001, 0.00001),
                                                                          (-0.000001, 0.00001),
                                                                          True, (0.7, 1.3), 'constant', 0, 3,
                                                                          'constant',
                                                                          0, 0,
                                                                          random_crop=False)(mov_img, mov_label,
                                                                                             fix_img, fix_label)

            mov_img = limit(mov_img)
            mov_img = Nor(mov_img)
            mov_img = np.reshape(mov_img, (1,) + mov_img.shape + (1,))

            fix_img = limit(fix_img)
            fix_img = Nor(fix_img)
            fix_img = np.reshape(fix_img, (1,) + fix_img.shape + (1,))

            fix_label_3 = fix_label.astype('int16')
            d = np.where(fix_label_3 > 4)
            fix_label_3[d] = 0
            fix_label_3 = tf.keras.utils.to_categorical(fix_label_3, num_classes=5)
            fix_label_3 = fix_label_3[np.newaxis, :, :, :, :]

            mov_label_3 = mov_label.astype('int16')
            d = np.where(mov_label_3 > 4)
            mov_label_3[d] = 0
            mov_label_org = mov_label_3[np.newaxis, :, :, :, np.newaxis]
            mov_label_3 = tf.keras.utils.to_categorical(mov_label_3, num_classes=5)
            mov_label_3 = mov_label_3[np.newaxis, :, :, :, :]


            X_data += [mov_img]
            Y_data += [fix_img]
            R_data += [fix_label_3]
            S_data += [mov_label_3]
            F_data += [flag]
            O_data += [mov_label_org]
        if bs > 1:
            return_vals = [np.concatenate(X_data, 0)]
            return_vals.append(np.concatenate(Y_data, 0))
            return_vals.append(np.concatenate(R_data, 0))
            return_vals.append(np.concatenate(S_data, 0))
            return_vals.append(np.concatenate(F_data, 0))
            return_vals.append(np.concatenate(O_data, 0))

        else:
            return_vals = [X_data[0]]
            return_vals.append(Y_data[0])
            return_vals.append(R_data[0])
            return_vals.append(S_data[0])
            return_vals.append(F_data[0])
            return_vals.append(O_data[0])
        ids += 1
        print('第%d次取数据'%(ids))

        yield tuple(return_vals)


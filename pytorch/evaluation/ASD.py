from os import listdir

import numpy as np
from os.path import join
import SimpleITK as sitk


def is_image3d_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def HausdorffDistance(predict, label, index=1):
    predict = (predict == index).astype(np.uint8)
    label = (label == index).astype(np.uint8)
    predict_sum  =  predict.sum()
    label_sum = label.sum()
    if predict_sum != 0 and label_sum != 0 :
        mask1 = sitk.GetImageFromArray(predict,isVector=False)
        mask2 = sitk.GetImageFromArray(label,isVector=False)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(mask1, mask2)
        result1 = hausdorff_distance_filter.GetHausdorffDistance()
        result2 = hausdorff_distance_filter.GetAverageHausdorffDistance()
        result = result1,result2
    elif predict_sum != 0 and label_sum == 0:
        result = 'FP','FP'
    elif predict_sum == 0 and label_sum != 0:
        result = 'FN','FN'
    else:
        result = 'TN','TN'
    return result

def Getcontour(img):

    image = sitk.GetImageFromArray(img.astype(np.uint8),isVector=False)

    filter = sitk.SimpleContourExtractorImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image)
    return image.astype(np.uint8)

def ASD(results_dir, model_name, n_classes, pred_dir, gt_dir):
    image_filenames = listdir(join(results_dir, model_name, 'fl'))
    hauAve = np.zeros(shape=(n_classes, len(image_filenames)), dtype=np.float32)
    hau = np.zeros(shape=(n_classes, len(image_filenames)), dtype=np.float32)
    i = 0
    for imagename in image_filenames:
        predict = np.fromfile(join(results_dir, model_name, pred_dir, imagename), dtype=np.float32)
        predict = predict.reshape((144, 144, 128))
        predict = to_categorical(predict, num_classes=n_classes)

        groundtruth = np.fromfile(join(results_dir, model_name, gt_dir, imagename), dtype=np.float32)
        groundtruth = groundtruth.reshape((144, 144, 128))
        groundtruth = to_categorical(groundtruth, num_classes=n_classes)

        tmp = 1
        for j in range(n_classes):
            predict_suf = Getcontour(predict[j])
            label_suf = Getcontour(groundtruth[j])
            a = HausdorffDistance(predict_suf, label_suf)
            if a[0] == 'FN':
                print(a)
                tmp = 0
            else:
                hau[j, i], hauAve[j, i] = a[0], a[1]
        print(imagename, hau[:, i], hauAve[:, i])

        if tmp == 1:
            i += 1

    return hau[1:, 0:i], hauAve[1:, 0:i]

if __name__ == '__main__':
    hau, hauAve = ASD('results', "pcreg_5shot", 8, 's_m', 'ml')
    mean_hau = np.mean(hau, axis=0)
    mean_hauAve = np.mean(hauAve, axis=0)

    std_hau = np.std(mean_hau, axis=0)
    std_hauAve = np.std(mean_hauAve, axis=0)

    mean_hau = np.mean(mean_hau, axis=0)
    mean_hauAve = np.mean(mean_hauAve, axis=0)
    print(mean_hau, std_hau, mean_hauAve, std_hauAve)

import numpy as np
import torch
import SimpleITK as sitk

def GetSD(predict,label):
    predict = predict.astype(np.uint8)
    label = label.astype(np.uint8)
    mask1 = sitk.GetImageFromArray(predict,isVector=False)
    mask2 = sitk.GetImageFromArray(label,isVector=False)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(mask1, mask2)
    ave_distance = hausdorff_distance_filter.GetAverageHausdorffDistance()
    idx_predict = np.where(predict!=0)
    sum=0
    print(np.size(idx_predict[0]))
    for i in range(np.size(idx_predict[0])):
        mask_temp = np.zeros_like(predict,dtype=np.uint8)
        mask_temp[idx_predict[0][i]][idx_predict[1][i]][idx_predict[2][i]]=1
        mask_temp = sitk.GetImageFromArray(mask_temp,isVector=False)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(mask_temp, mask2)
        distance_pixel = hausdorff_distance_filter.GetHausdorffDistance()
        sum=sum+np.square(distance_pixel-ave_distance)
    result=np.sqrt(sum/np.size(idx_predict[0]))
    return result

def Getcontour(img):

    image = sitk.GetImageFromArray(img.astype(np.uint8),isVector=False)

    filter = sitk.SimpleContourExtractorImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image)
    return image.astype(np.uint8)

def GetMaxConponent(img, index=1):

    if img.max() < index:
        return np.zeros_like(img,dtype=np.uint8)

    image = sitk.GetImageFromArray((img == index).astype(np.uint8),isVector=False)

    filter = sitk.ConnectedComponentImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image).astype(np.uint8)
    maxindex = 0
    max_sum = 0
    for i in range(1, image.max()+1):
        temp = (image == i).sum()
        if temp > max_sum:
            max_sum = temp
            maxindex = i

    if maxindex == 0:
        return np.zeros_like(img, dtype=np.uint8)
    else:
        return (image == maxindex).astype(np.uint8) * index


def GrayMorphologicalClosingImage(img):
    image = sitk.GetImageFromArray(img.astype(np.uint8),isVector=False)

    filter = sitk.GrayscaleMorphologicalClosingImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image)
    return image.astype(np.uint8)

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


def dice3D(eval_segm, gt_segm, index=1):
    '''
    eval_segm:the matirx to evaluate
    gt_segm:  ground truth

    '''
    if type(eval_segm) == np.ndarray:
        eval_segm = torch.from_numpy(eval_segm).byte()
    if type(gt_segm) == np.ndarray:
        gt_segm = torch.from_numpy(gt_segm).byte()

    eps = 1e-6
    #assert eval_segm.size == gt_segm.size
    #gt_segm = gt_segm.byte()
    eval_segm = (eval_segm == index)

    sum_eval = eval_segm.sum().item()

    gt_segm = (gt_segm == index)

    sum_gt = gt_segm.sum().item()

    if sum_eval != 0 and sum_gt != 0:
        intersection = torch.sum(eval_segm * gt_segm).item()
        union = torch.sum(eval_segm).item() + torch.sum(gt_segm).item() + eps
        dice_ = 2.0 * intersection / union
    elif sum_eval != 0 and sum_gt == 0:
        dice_ = 'FP'
    elif sum_eval == 0 and sum_gt != 0:
        dice_ = 'FN'
    else:
        dice_ = 'TN'

    return dice_


def jaccard(eval_segm, gt_segm, index=1):
    '''
    eval_segm:the matirx to evaluate
    gt_segm:  ground truth

    '''
    if type(eval_segm) == np.ndarray:
        eval_segm = torch.from_numpy(eval_segm.copy()).byte()
    if type(gt_segm) == np.ndarray:
        gt_segm = torch.from_numpy(gt_segm.copy()).byte()

    eps = 1e-6
    #assert eval_segm.size == gt_segm.size
    #gt_segm = gt_segm.byte()
    eval_segm[eval_segm != index] = 0
    eval_segm[eval_segm == index] = 1
    sum_eval = eval_segm.sum().item()
    gt_segm[gt_segm != index] = 0
    gt_segm[gt_segm == index] = 1

    sum_gt = gt_segm.sum().item()

    if sum_eval != 0 and sum_gt != 0:
        intersection = torch.sum(eval_segm * gt_segm).item()
        union = torch.sum(eval_segm).item() + torch.sum(gt_segm).item() - intersection + eps
        dice_ = intersection / union
    elif sum_eval != 0 and sum_gt == 0:
        dice_ = 'FP'
    elif sum_eval == 0 and sum_gt != 0:
        dice_ = 'FN'
    else:
        dice_ = 'TN'

    return dice_

def pixel_accuracy_ex(eval_segm, gt_segm):
    '''
    eval_segm,gt_segm should be format of (N_slice,height,width)
    '''
    assert (eval_segm.shape == gt_segm.shape)
    num = eval_segm.shape[0]

    result = np.zeros((num), np.float32)

    for i in range(num):
        result[i] = pixel_accuracy(eval_segm[i, ...], gt_segm[i, ...])

    return result.mean()


def mean_accuracy_ex(eval_segm, gt_segm):
    '''
    eval_segm,gt_segm should be format of (N_slice,height,width)
    '''
    assert(eval_segm.shape == gt_segm.shape)
    num = eval_segm.shape[0]

    result = np.zeros((num), np.float32)

    for i in range(num):
        result[i] = mean_accuracy(eval_segm[i, ...], gt_segm[i, ...])

    return result.mean()


def mean_IU_ex(eval_segm, gt_segm):
    '''
    eval_segm,gt_segm should be format of (N_slice,height,width)
    '''
    assert (eval_segm.shape == gt_segm.shape)
    num = eval_segm.shape[0]

    result = np.zeros((num), np.float32)

    for i in range(num):
        result[i] = mean_IU(eval_segm[i, ...], gt_segm[i, ...])

    return result.mean()


def frequency_weighted_IU_ex(eval_segm, gt_segm):
    '''
    eval_segm,gt_segm should be format of (N_slice,height,width)
    '''
    assert (eval_segm.shape == gt_segm.shape)
    num = eval_segm.shape[0]

    result = np.zeros((num), np.float32)

    for i in range(num):
        result[i] = frequency_weighted_IU(eval_segm[i, ...], gt_segm[i, ...])

    return result.mean()


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, ...]
        curr_gt_mask = gt_mask[i, ...]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def check_size(eval_segm, gt_segm):
    assert eval_segm.shape == gt_segm.shape


def extract_masks(segm, cl, n_cl):
    slices, h, w = segm.shape
    masks = np.zeros((n_cl, slices, h, w))

    for i, c in enumerate(cl):
        masks[i, ...] = segm == c

    return masks


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

import os
from os import listdir
from os.path import join
import numpy as np
import SimpleITK as sitk
from utils.utils import to_categorical, dice

def Get_Ja(displacement):
    D_y = (displacement[:, 1:, :-1, :-1] - displacement[:, :-1, :-1, :-1])

    D_x = (displacement[:, :-1, 1:, :-1] - displacement[:, :-1, :-1, :-1])

    D_z = (displacement[:, :-1, :-1, 1:] - displacement[:, :-1, :-1, :-1])

    D1 = (D_x[0] + 1) * ((D_y[1] + 1) * (D_z[2] + 1) - D_z[1] * D_y[2])

    D2 = (D_x[1]) * (D_y[0] * (D_z[2] + 1) - D_y[2] * D_x[0])

    D3 = (D_x[2]) * (D_y[0] * D_z[1] - (D_y[1] + 1) * D_z[0])
    return D1 - D2 + D3

def Ja(results_dir, model_name, flow_dir):
    image_filenames = listdir(join(results_dir, model_name, flow_dir))
    ja = np.zeros(len(image_filenames))

    for i in range(len(image_filenames)):
        name = image_filenames[i]
        flow = np.fromfile(join(results_dir, model_name, flow_dir, name), dtype=np.float32)
        flow = flow.reshape((3, 144, 144, 128))
        flow = flow.transpose(1, 2, 3, 0)
        # a = Get_Ja(flow)
        # count = len(np.where(a <= 0)[0])

        flow = sitk.GetImageFromArray(flow)
        ja1 = sitk.DisplacementFieldJacobianDeterminant(flow)
        ja1 = sitk.GetArrayFromImage(ja1)
        count = np.where(ja1 <= 0, 1, 0)
        ja[i] = np.sum(count)/(np.sum(np.ones_like(count)))
        # a = ja1[warp_mask > 0]
        # b = a[a <= 0]
        # count1 = len(b)
        # ja[i] = Get_Ja(flow)

        print(name, ja[i])
    print(np.mean(ja), np.std(ja))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Ja('results', "pcreg_5shot", 'flow')
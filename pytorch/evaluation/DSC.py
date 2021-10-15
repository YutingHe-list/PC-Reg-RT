import os
from os import listdir
from os.path import join

from utils.utils import to_categorical, dice
import numpy as np



def DSC(results_dir, model_name, n_classes, pred_dir, gt_dir):
    image_filenames = listdir(join(results_dir, model_name, 'fl'))
    DSC = np.zeros((n_classes, len(image_filenames)))

    for i in range(len(image_filenames)):
        name = image_filenames[i]
        fl = np.fromfile(join(results_dir, model_name, gt_dir, name), dtype=np.float32)
        fl = to_categorical(fl, n_classes)
        ml = np.fromfile(join(results_dir, model_name, pred_dir, name), dtype=np.float32)
        ml = to_categorical(ml, n_classes)

        for c in range(n_classes):
            DSC[c, i] = dice(ml[c], fl[c])

        print(name, DSC[1:, i])

    print(np.mean(DSC[1:, :], axis=1))
    print(np.mean(DSC[1:, :]), np.std(np.mean(DSC[1:, :], axis=0)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DSC('results', "pcreg_5shot", 8, 'w_label_m_to_f', 'fl')




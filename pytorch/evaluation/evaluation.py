#!/usr/bin/env python
import argparse
from scipy import io as sio
import os
import glob
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

def tostr(x):
    if type(x) == float or  type(x) == np.float:
        x = '%0.4f' % (x)
    return x

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--matpath', required=True, help=' the segment result')

    args = parser.parse_args()

    assert os.path.exists(args.matpath)

    files = glob.glob(args.matpath + "/*.mat")

    dstpath = os.path.join(args.matpath, '..', 'evaluation_all.csv')
    all_result_path = os.path.join(args.matpath, '..', 'result_all')
    if not os.path.exists(all_result_path):
        os.mkdir(all_result_path)

    fid = open(dstpath, 'w')
    index = [
             'mean_IOU_direct','kidney_dice_direct','tumor_dice_direct','kidney_hau_direct','tumor_hau_direct','kidney_hauAve_direct','tumor_hauAve_direct',
             'mean_IOU_Morp', 'kidney_dice_Morp', 'tumor_dice_Morp', 'kidney_hau_Morp', 'tumor_hau_Morp','kidney_hauAve_Morp', 'tumor_hauAve_Morp',
             'mean_IOU_Morp_MaxCon', 'kidney_dice_Morp_MaxCon', 'tumor_dice_Morp_MaxCon', 'kidney_hau_Morp_MaxCon',
             'tumor_hau_Morp_MaxCon','kidney_hauAve_Morp_MaxCon','tumor_hauAve_Morp_MaxCon',
             'mean_IOU_MaxCon', 'kidney_dice_MaxCon', 'tumor_dice_MaxCon', 'kidney_hau_MaxCon', 'tumor_hau_MaxCon','kidney_hauAve_MaxCon', 'tumor_hauAve_MaxCon',
             'mean_IOU_MaxCon_Morp', 'kidney_dice_MaxCon_Morp', 'tumor_dice_MaxCon_Morp', 'kidney_hau_MaxCon_Morp',
             'tumor_hau_MaxCon_Morp','kidney_hauAve_MaxCon_Morp','tumor_hauAve_MaxCon_Morp',

             ]
    fid.write('filename,'+','.join(index)+'\n')
    for el in files:
        filename = el.split('/')[-1]
        mat = sio.loadmat(el)
        label = mat['label']
        predict = mat['predict']

		#这个是直接计算的
        kidney_dice_direct = dice3D(predict, label)
        tumor_dice_direct = dice3D(predict, label, 2)
        mean_IOU_direct = mean_IU(predict, label)
        kidney_hau_direct,kidney_hauAve_direct = HausdorffDistance(predict, label)
        tumor_hau_direct,tumor_hauAve_direct = HausdorffDistance(predict, label, 2)

		#这个好像是做了些形态学可以不管
        predict_Morp = GrayMorphologicalClosingImage(predict)


        predict_temp = predict_Morp.copy()
        predict_temp[predict_temp>0] = 1
        predict_temp = GetMaxConponent(predict_temp)
        predict_Morp = predict_Morp*predict_temp
        predict_Morp_MaxCon = GetMaxConponent(predict_Morp) + GetMaxConponent(predict_Morp,index=2)

        predict_temp = predict.copy()
        predict_temp[predict_temp>0] = 1
        predict_temp = GetMaxConponent(predict_temp)
        predict_MaxCon = predict*predict_temp

		#这个做了最大连通域，排除了一些误分割的结果
        predict_MaxCon_kid = GetMaxConponent(predict_MaxCon)
        predict_MaxCon_tum =  GetMaxConponent(predict_MaxCon,index=2)
        predict_MaxCon_tum[np.where(predict_MaxCon_tum!=0)]=1
        label_kid = GetMaxConponent(label)
        label_tum =  GetMaxConponent(label,index=2)
        label_tum[np.where(label_tum!=0)]=1

        predict_MaxCon = GetMaxConponent(predict_MaxCon) + GetMaxConponent(predict_MaxCon,index=2)

		#这些提取了表面
        predict_MaxCon_suf_kid = Getcontour(predict_MaxCon_kid) 
        predict_MaxCon_suf_tum = Getcontour(predict_MaxCon_tum)
        predict_MaxCon_suf_tum[np.where(predict_MaxCon_suf_tum!=0)]=2
        label_suf_kid = Getcontour(label_kid)
        label_suf_tum = Getcontour(label_tum)
        label_suf_tum[np.where(label_suf_tum!=0)]=2


        kidney_dice_MaxCon = dice3D(predict_MaxCon, label)
        tumor_dice_MaxCon = dice3D(predict_MaxCon, label, 2)
        mean_IOU_MaxCon = mean_IU(predict_MaxCon, label)
        kidney_hau_MaxCon,kidney_hauAve_MaxCon = HausdorffDistance(predict_MaxCon_suf_kid, label_suf_kid)
        tumor_hau_MaxCon,tumor_hauAve_MaxCon = HausdorffDistance(predict_MaxCon_suf_tum, label_suf_tum, 2)

        predict_MaxCon_Morp = GrayMorphologicalClosingImage(predict_MaxCon)
        kidney_dice_MaxCon_Morp = dice3D(predict_MaxCon_Morp, label)
        tumor_dice_MaxCon_Morp = dice3D(predict_MaxCon_Morp, label, 2)
        mean_IOU_MaxCon_Morp = mean_IU(predict_MaxCon_Morp, label)
        kidney_hau_MaxCon_Morp,kidney_hauAve_MaxCon_Morp = HausdorffDistance(predict_MaxCon_Morp, label)
        tumor_hau_MaxCon_Morp,tumor_hauAve_MaxCon_Morp = HausdorffDistance(predict_MaxCon_Morp, label, 2)


        result = ''
        for el in index:
            result += '%s,'%(tostr(eval(el)))

        line = '%s,%s\n' % (filename, result[0:-1])
        mdict = {'predict_direct':predict,'predict_Morp':predict_Morp,'predict_Morp_MaxCon':predict_Morp_MaxCon,
                 'predict_MaxCon_kid':predict_MaxCon_suf_kid,'predict_MaxCon_tum':predict_MaxCon_suf_tum,
                 'predict_MaxCon_Morp':predict_MaxCon_Morp,'label':label}
        sio.savemat(os.path.join(all_result_path,filename),mdict)
        fid.write(line)

    fid.close()

    dstpath = os.path.abspath(dstpath)
    cmd = 'python ' + os.path.abspath(os.path.join(os.path.dirname(__file__),'to_excel.py')) +' --csv_file ' + dstpath
    os.system(cmd)

if __name__ == "__main__":
    main()

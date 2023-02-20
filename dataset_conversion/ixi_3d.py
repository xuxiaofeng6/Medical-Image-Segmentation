import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import random
import yaml
import copy
import pdb

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    
    imLabel.CopyInformation(imImage)

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))

    imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    

    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 10, 10])

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = ''
    lab_path = ''
    tgt_path = ''

    pathExist(tgt_path)
    
    name_list = []
    for idx,case in enumerate(os.listdir(lab_path)):
        i = case.split('.')[0]
        name_list.append(i)

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for index,name in enumerate(name_list):
        if index >= 0:
            print(index,name)
            # img_name = '%d.nii.gz'%name
            # lab_name = '%d.nii.gz'%name

            img_name = name + '.nii.gz'

            img = sitk.ReadImage(src_path + img_name)
            lab = sitk.ReadImage(lab_path + img_name)

            # ResampleImage(img, lab, tgt_path, name, (0.767578125, 0.767578125, 1.0))
            ResampleImage(img, lab, tgt_path, name, (1, 1, 1))
            print(name, 'done')



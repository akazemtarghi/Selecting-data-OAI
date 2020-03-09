import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from data_processing_utils import read_dicom,\
    process_xray, read_pts, extract_patches,f_2_b, pad,worker
import argparse
import glob
from scipy.ndimage import interpolation
from PIL import Image
import matplotlib
from read_json import Read_json
from standard_ROI import standard_ROI_amir, show_patches
from polygan import polylines_amir
from data_processing_utils import rotate, rotate_amir

output_123_oa = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/output_123_oa.csv')
output_123_oa = output_123_oa.loc[(output_123_oa['SeriesDescription'] != 'Bilateral PA Fixed Flexion Knee')]
output_123_oa.reset_index(inplace=True)
list_ID = output_123_oa[['ParticipantID']+['SeriesDescription']].copy()


# can be changed in accordance with subgroups
prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/New folder (2)/'
dir_t = '/t.json'
dir_f = '/f.json'



for i in range(len(list_ID)):


    try:

        if list_ID['ParticipantID'][i] == 9401202:
            a = 1


        image_dir_f = '/' + list_ID['SeriesDescription'][i] + '_femur_.png'
        image_dir_t = '/' + list_ID['SeriesDescription'][i] + '_tibia_.png'

        list_x_tibia, list_y_tibia = Read_json(prefix + str(list_ID['ParticipantID'][i]) + dir_t)
        list_x_femur, list_y_femur = Read_json(prefix + str(list_ID['ParticipantID'][i]) + dir_f)

        mri_image_tibia = cv2.imread(prefix + str(list_ID['ParticipantID'][i]) + image_dir_t, 0)
        mri_image_femur = cv2.imread(prefix + str(list_ID['ParticipantID'][i]) + image_dir_f, 0)

        # determining the center of joint
        x = (list_x_tibia[8] + list_x_tibia[9])//2
        y = (list_y_tibia[8] + list_y_tibia[9]) // 2

        center = np.array([x, y])
        p1 = np.array([list_x_tibia[5], list_y_tibia[5]])
        p2 = np.array([list_x_tibia[12], list_y_tibia[12]])

        points_t = np.zeros([18, 2])
        points_t[:, 0] = list_x_tibia
        points_t[:, 1] = list_y_tibia

        points_f = np.zeros([17, 2])
        points_f[:, 0] = list_x_femur
        points_f[:, 1] = list_y_femur

        rotated_image_tibia, M = rotate(mri_image_tibia, center, p1, p2)
        rotated_landmarks = rotate_amir(points_t, M)

        # plt.figure()
        # plt.imshow(mri_image_tibia)
        #
        # plt.figure()
        # plt.imshow(rotated_image_tibia)
        # plt.scatter(x=rotated_landmarks[:, 0],y=rotated_landmarks[:, 1])









        image_mask_tibia = polylines_amir(rotated_landmarks, rotated_image_tibia)
        #image_mask_femur = polylines_amir(points_t, mri_image_femur)

        #
        # plt.figure()
        # plt.imshow(image_mask_tibia)


        roi_med_t, roi_lat_t, r, box_med_t, box_lat_t = standard_ROI_amir(image_mask_tibia, rotated_landmarks[:, 0], rotated_landmarks[:, 1])
        #roi1, roi2, r, box1, box2 = standard_ROI_amir(image_mask_femur, list_x_femur, list_y_femur)

        show_patches(rotated_image_tibia, box_lat_t, box_med_t, i, r)





    except:

        continue



    if (len(roi_lat_t[roi_lat_t==0]) < 5) & (len(roi_med_t[roi_med_t == 0]) < 5):
        a = prefix + output_123_oa['SeriesDescription'][i] + '_tibia_' + 'roi_lat_t' + '.png'

        roi_med_t, roi_lat_t, r, box_med_t, box_lat_t = standard_ROI_amir(rotated_image_tibia, rotated_landmarks[:, 0],  rotated_landmarks[:, 1])

        matplotlib.image.imsave(prefix + str(list_ID['ParticipantID'][i]) + '/roi_lat_' + list_ID['SeriesDescription'][i] + '_tibia_.png',
                                roi_lat_t,
                                cmap=plt.cm.bone)
        matplotlib.image.imsave(prefix + str(list_ID['ParticipantID'][i]) + '/roi_med_' + list_ID['SeriesDescription'][i] + '_tibia_.png',
                                roi_med_t,
                                cmap=plt.cm.bone)

    else:
        #show_patches(mri_image_tibia, box_lat_t, box_med_t, i, r)
        print('Not correct lateral')
        print(list_ID['ParticipantID'][i])
        print(i)





















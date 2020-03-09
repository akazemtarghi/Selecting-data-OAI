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
from rectangle_growing import rec_growing, show_patches_femur

output_123_oa = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/output_123_oa.csv')
output_123_oa = output_123_oa.loc[(output_123_oa['SeriesDescription'] != 'Bilateral PA Fixed Flexion Knee')]
output_123_oa.reset_index(inplace=True)
list_ID = output_123_oa[['ParticipantID']+['SeriesDescription']].copy()


# can be changed in accordance with subgroups
prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/New folder (2)/'
dir_t = '/t.json'
dir_f = '/f.json'



for i in range(90, len(list_ID)):


    try:

        if list_ID['ParticipantID'][i] == 9252130:
            a = 1


        image_dir_f = '/' + list_ID['SeriesDescription'][i] + '_femur_.png'
        image_dir_t = '/' + list_ID['SeriesDescription'][i] + '_tibia_.png'

        list_x_tibia, list_y_tibia = Read_json(prefix + str(list_ID['ParticipantID'][i]) + dir_t)
        list_x_femur, list_y_femur = Read_json(prefix + str(list_ID['ParticipantID'][i]) + dir_f)

        mri_image_tibia = cv2.imread(prefix + str(list_ID['ParticipantID'][i]) + image_dir_t, 0)
        mri_image_femur = cv2.imread(prefix + str(list_ID['ParticipantID'][i]) + image_dir_f, 0)

    except:
        continue

    try:

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

        rotated_image_femur, M = rotate(mri_image_femur, center, p1, p2)
        rotated_landmarks = rotate_amir(points_f, M)

        mask_femur = polylines_amir(rotated_landmarks, rotated_image_femur)



        rec_m, width_m, length_m, x_m, y_m = rec_growing(mask_femur, rotated_landmarks,side='medial', name='femur')
        rec_m = mask_femur[y_m - width_m:y_m, x_m - length_m: x_m]
        rec_lat, width_lat, length_lat, x_lat, y_lat = rec_growing(mask_femur, rotated_landmarks,side='lateral', name='femur')

        # if width_lat > width_m:
        #
        #     width_lat = width_m
        #     length_lat = length_m
        #
        # else:
        #
        #     width_m = width_lat
        #     length_m = length_lat





            # diff1 = width_lat - width_m
            # diff2 = length_lat - length_m
            # width_lat = width_lat - diff1
            # length_lat = length_lat - diff2

        rec_lat = mask_femur[y_lat - width_lat:y_lat, x_lat - length_lat: x_lat]
        show_patches_femur(mask_femur, width_m, length_m, x_m, y_m, i, name='medial')
        show_patches_femur(mask_femur, width_lat, length_lat, x_lat, y_lat, i, name='lateral')

    except:

        print(list_ID['ParticipantID'][i])












        # plt.figure()
        # plt.imshow(mri_image_tibia)
        #
        # plt.figure()
        # plt.imshow(rotated_image_tibia)
        # plt.scatter(x=rotated_landmarks[:, 0],y=rotated_landmarks[:, 1])



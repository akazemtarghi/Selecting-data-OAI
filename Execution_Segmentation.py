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

def Execution_Segmentation_amir(list_ID):

    dir_t = '/via_export_json (1).json'
    dir_f = '/via_export_json.json'

    if list_ID['Label'] == 1:
        prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/OA/'
    else:
        prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/No-OA/'




    image_dir_f = '/' + list_ID['SeriesDescription'] + '_femur_.png'
    image_dir_t = '/' + list_ID['SeriesDescription'] + '_tibia_.png'

    list_x_tibia, list_y_tibia = Read_json(prefix + str(list_ID['ParticipantID']) + dir_t)
    list_x_femur, list_y_femur = Read_json(prefix + str(list_ID['ParticipantID']) + dir_f)

    mri_image_tibia = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t, 0)
    mri_image_femur = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_f, 0)

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
    rotated_image_femur, M = rotate(mri_image_femur, center, p1, p2)

    rotated_landmarks_t = rotate_amir(points_t, M)
    rotated_landmarks_f = rotate_amir(points_f, M)

    image_mask_tibia = polylines_amir(rotated_landmarks_t, rotated_image_tibia)
    # image_mask_femur = polylines_amir(rotated_landmarks_f, mri_image_femur)

    roi_med_t, roi_lat_t, r, box_med_t, box_lat_t = standard_ROI_amir(image_mask_tibia,
                                                                      rotated_landmarks_t[:, 0],
                                                                      rotated_landmarks_t[:, 1])

    # roi_med_f, roi_lat_f, r, box_med_f, box_lat_f = standard_ROI_amir(image_mask_femur,
    #                                                                   rotated_landmarks_f[:, 0],
    #                                                                   rotated_landmarks_f[:, 1])
    #





    if (len(roi_lat_t[roi_lat_t==0]) < 5) & (len(roi_med_t[roi_med_t == 0]) < 5):

        roi_med_t, roi_lat_t, r, box_med_t, box_lat_t = standard_ROI_amir(mri_image_tibia,
                                                                          rotated_landmarks_t[:, 0],
                                                                          rotated_landmarks_t[:, 1])

        return roi_med_t, roi_lat_t

    else:
        #show_patches(mri_image_tibia, box_lat_t, box_med_t, i, r)
        print('Not correct lateral')
        print(list_ID['ParticipantID'])






















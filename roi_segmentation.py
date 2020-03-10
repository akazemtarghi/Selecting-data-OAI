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
from standard_ROI import standard_ROI_amir, show_patches, standard_ROI_amir_femur
from polygan import polylines_amir
from data_processing_utils import rotate, rotate_amir
import matplotlib.pyplot as plt

csv_file = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/ALL.csv')

csv_file.drop_duplicates(subset=['ParticipantID', 'SeriesDescription'], inplace=True)
csv_file.reset_index(inplace=True)
csv_file = csv_file.drop(columns=['index', 'level_0'])
list_ID = csv_file.copy()

# can be changed in accordance with subgroups
dir_t = '/via_export_json (1).json'
dir_f = '/via_export_json.json'
aa = list_ID['Label'].loc[0]


for i in range(len(csv_file)):
    list_ID = csv_file.loc[i]




    # if list_ID['ParticipantID'][i] == 9401202:
    #     a = 1

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
    rotated_landmarks = rotate_amir(points_t, M)

    rotated_mri_femur, M = rotate(mri_image_femur, center, p1, p2)
    rotated_landmarks_f = rotate_amir(points_f, M)

    image_mask_tibia = polylines_amir(rotated_landmarks, rotated_image_tibia)
    #image_mask_femur = polylines_amir(points_t, mri_image_femur)

    roi_med_t, roi_lat_t, r, box_med_t, box_lat_t = standard_ROI_amir(image_mask_tibia,
                                                                      rotated_landmarks[:, 0],
                                                                      rotated_landmarks[:, 1])

    roi1, roi2, r, coor_m, coor_l = standard_ROI_amir_femur(rotated_mri_femur,
                                                            rotated_landmarks_f[:, 0],
                                                            rotated_landmarks_f[:, 1])

    show_patches(rotated_mri_femur, r, coor_m, coor_l, i)
    plt.close('all')
























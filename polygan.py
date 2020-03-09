import numpy as np
import cv2
from read_json import Read_json
import glob
import matplotlib.pyplot as plt

def polylines_amir(points, image):

    temp = image.copy()


    points = points.reshape((-1, 1, 2))

    temp = cv2.fillPoly(temp, np.int32([points]), 255).copy()
    temp[temp != 255] = 0
    temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel=(10, 10)).copy()

    return temp


def get_landmarks(prefix, dir_tibia, dir_femur):

    x_list_tibia = []
    y_list_tibia = []
    x_femur_list = []
    y_femur_list = []

    image_list = []

    for filename in glob.glob(prefix + '/*'):
        a = filename + dir_tibia
        # for tibia
        x_tibia, y_tibia = Read_json(filename + dir_tibia)
        # for femur
        x_femur, y_femur = Read_json(filename + dir_femur)

        x_femur_list.append(x_femur)
        y_femur_list.append(y_femur)

        x_list_tibia.append(x_tibia)
        y_list_tibia.append(y_tibia)

        for filename1 in glob.glob(filename + '/*.png'):
            image = cv2.imread(filename1, 0)
            image_list.append(image)

    return image_list, x_list_tibia, y_list_tibia, x_femur_list, y_femur_list




# prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/New folder (2)'
# dir_tibia = '/t.json'
# dir_femur = '/f.json'
#
# image_list, x_list_tibia, y_list_tibia, x_femur_list, y_femur_list = get_landmarks(prefix, dir_tibia, dir_femur)
#
#
# I1 = image_list[1]
# I2 = I1.copy()
#
# points_t = np.zeros([18, 2])
# points_t[:, 0] = x_list_tibia[0]
# points_t[:, 1] = y_list_tibia[0]
#
#
# points_f = np.zeros([17, 2])
# points_f[:, 0] = x_femur_list[0]
# points_f[:, 1] = y_femur_list[0]
#
# image_mask_tibia = polylines_amir(points_t, I1)
#
# plt.figure()
# plt.imshow(image_mask_tibia, cmap=plt.cm.bone)
# image_mask_femur = polylines_amir(points_f, I2)
#
# plt.figure()
# plt.imshow(image_mask_femur, cmap=plt.cm.bone)



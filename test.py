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


parser = argparse.ArgumentParser()
parser.add_argument('--oai_root', default='/data/OAI_xrays')
parser.add_argument('--landmarks_root', default='landmark_annotations_BF_OAI')
parser.add_argument('--save_results', default='patches/')
parser.add_argument('--pad', type=int, default=400)
parser.add_argument('--spacing', type=float, default=0.2)
parser.add_argument('--sizemm', type=int, default=140)
parser.add_argument('--o_sizemm_w', type=int, default=25)
parser.add_argument('--o_sizemm_h', type=int, default=25)
parser.add_argument('--n_threads', type=int, default=140)
args = parser.parse_args()
slices = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/coordinates2.csv',
                     names=['ID', 'folder', "tibia", "femur"])

output_250_hl = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/output_123_oa.csv')

x = output_250_hl.drop_duplicates(subset=['ParticipantID'])
#output_123_oa = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/output_123_oa.csv')

prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/OAI_Amir/00m/'
prefix_land = 'C:/Users/Amir Kazemtarghi/Documents/data/desire_landmark/'
prefix_save = 'C:/Users/Amir Kazemtarghi/Documents/data/5slice/'
import scipy.ndimage


def taking_slices_out(img3d, num1, num2, flip=False):

    mri_image_tibia = img3d[:, num1, :]
    mri_image_tibia = interpolation.zoom(mri_image_tibia, [1, 1/sag_aspect], mode='wrap')

    if flip:
        mri_image_tibia = cv2.flip(mri_image_tibia, 1)

    #mri_image_tibia = mri_image_tibia.astype('uint8')
    #mri_image_tibia = 255 - mri_image_tibia

    mri_image_femur = img3d[:, num2, :]
    mri_image_femur = interpolation.zoom(mri_image_femur, [1, 1 / sag_aspect], mode='wrap')

    if flip:
        mri_image_femur = cv2.flip(mri_image_femur, 1)

    #mri_image_femur = mri_image_femur.astype('uint8')
    #mri_image_femur = 255 - mri_image_femur

    return mri_image_tibia, mri_image_femur



def resample(image, ss,ps, new_spacing=[0.2, 0.2]):

    # Determine current pixel spacing
    spacing = map(float, ([ps[1]]+ [ss]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def boxing(box, mri_image):
    if (box % 2) == 0:
        initial = box/2
        final_mri = mri_image[50:(mri_image.shape[0]-27), :]
    else:
        initial = box//2
        initial2 = initial +1
        final_mri = mri_image[50:(mri_image.shape[0] - 27), :]

    return final_mri





def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

p = 0

for i in range(len(output_250_hl)):

    #print(output_250_hl['ParticipantID'][i])

    if output_250_hl['SeriesDescription'][i] == 'Bilateral PA Fixed Flexion Knee':

        dicomimage = prefix + output_250_hl['Folder'][i]+ '/001'
        pathsave = prefix_save + str(output_250_hl['ParticipantID'][i]) + '/'
        land_address1 = prefix_land + output_250_hl['Folder'][i] + '/001.pts'
        land_address2 = prefix_land + output_250_hl['Folder'][i] + '/001_f.pts'
        job_data = [args,    dicomimage, land_address1, land_address2, pathsave]
        a, dim_R, dim_L, L, R = worker(job_data)
        if a == 1:
            print('something is wrong')

    else:

        image_list = []
        for filename in glob.glob(prefix + output_250_hl['Folder'][i] + '/*'):  # assuming gif
            im = pydicom.dcmread(filename)
            image_list.append(im)

        image_list = sorted(image_list, key=lambda s: s.SliceLocation)

        # pixel aspects, assuming all slices are the same
        ps = image_list[0].PixelSpacing
        ss = image_list[0].SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]

        # create 3D array
        img_shape = list(image_list[0].pixel_array.shape)
        img_shape.append(len(image_list))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for k, s in enumerate(image_list):
            img2d = s.pixel_array
            img3d[:, :, k] = img2d

        num1 = int(slices['tibia'][p])
        num2 = int(slices['femur'][p])

        if output_250_hl['SeriesDescription'][i] == 'SAG_3D_DESS_LEFT':


            #print('LEFT')

            C_tibia, C_femur = taking_slices_out(img3d, num1, num2, flip=True)

            N_tibia, N_femur = taking_slices_out(img3d, num1 + 1, num2 + 1, flip=True)

            NN_tibia, NN_femur = taking_slices_out(img3d, num1 + 2, num2 + 2, flip=True)

            P_tibia, P_femur = taking_slices_out(img3d, num1 - 1, num2 - 1, flip=True)

            PP_tibia, PP_femur = taking_slices_out(img3d, num1 - 2, num2 - 2, flip=True)

            dim = (dim_L, dim_L)


        else:

            #print('Right')

            C_tibia, C_femur = taking_slices_out(img3d, num1, num2)

            N_tibia, N_femur = taking_slices_out(img3d, num1 + 1, num2 + 1)

            NN_tibia, NN_femur = taking_slices_out(img3d, num1 + 2, num2 + 2)

            P_tibia, P_femur = taking_slices_out(img3d, num1 - 1, num2 - 1)

            PP_tibia, PP_femur = taking_slices_out(img3d, num1 - 2, num2 - 2)

            dim = (dim_R, dim_R)

        p = p + 1

        #final_mri_tibia = interpolation.zoom(mri_image_tibia, [1, 1/sag_aspect], mode='wrap')
        # box = mri_image.shape[0]-mri_image.shape[1]
        # final_mri = boxing(box, mri_image)

        # resized_mri_tibia = cv2.resize(final_mri, dim, interpolation=cv2.INTER_AREA)

        #final_mri_femur = interpolation.zoom(mri_image_femur, [1, 1 / sag_aspect], mode='wrap')
        # box = mri_image.shape[0] - mri_image.shape[1]
        # final_mri = boxing(box, mri_image)

        # resized_mri_femur = cv2.resize(final_mri, dim, interpolation=cv2.INTER_AREA)

        index = ['C', 'N', 'NN', 'p', 'PP']

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_tibia_' + '_C' + '.png',
                                C_tibia,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_femur_' + '_C' + '.png',
                                C_femur,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_tibia_' + '_N' + '.png',
                                N_tibia,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_femur_' + '_N' + '.png',
                                N_femur,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_tibia_' + 'NN' + '.png',
                                NN_tibia,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_femur_' + 'NN' + '.png',
                                NN_femur,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_tibia_' + 'p' + '.png',
                                P_tibia,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_femur_' + 'p' + '.png',
                                P_femur,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_tibia_' + 'PP' + '.png',
                                PP_tibia,
                                cmap=plt.cm.bone)

        matplotlib.image.imsave(pathsave + output_250_hl['SeriesDescription'][i] + '_femur_' + 'PP' + '.png',
                                PP_femur,
                                cmap=plt.cm.bone)




#
# # save np.load
# np_load_old = np.load
#
# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--oai_root', default='/data/OAI_xrays')
# parser.add_argument('--landmarks_root', default='landmark_annotations_BF_OAI')
# parser.add_argument('--save_results', default='patches/')
# parser.add_argument('--pad', type=int, default=400)
# parser.add_argument('--spacing', type=float, default=0.2)
# parser.add_argument('--sizemm', type=int, default=140)
# parser.add_argument('--o_sizemm_w', type=int, default=25)
# parser.add_argument('--o_sizemm_h', type=int, default=25)
# parser.add_argument('--n_threads', type=int, default=140)
# args = parser.parse_args()
#
# dicomimage = 'C:/Users/Amir Kazemtarghi/Documents/data/OAI_Amir/00m/0.C.2/9015363/20041123/00430703/001'
# pathsave = 'C:/Users/Amir Kazemtarghi/Documents/data/result_preprocessing_crop/1'
# fname = 'C:/Users/Amir Kazemtarghi/Documents/data/OAI_Amir/LANDMARKS/00/0.C.2/44/9015363/20041123/00430703/001.pts'
# fname2 = 'C:/Users/Amir Kazemtarghi/Documents/data/OAI_Amir/LANDMARKS/00/0.C.2/44/9015363/20041123/00430703/001_f.pts'
# job_data = [args, dicomimage, fname, fname2, pathsave]
#
# a = worker(job_data)
#
# ghgh = np.load('C:/Users/Amir Kazemtarghi/Documents/data/result_preprocessing_crop/1.npy')
#
# aaaa = ghgh[1]
# a2 = aaaa['R']
# a3 = a2['lat']
#
#
# plt.imshow(a3)
# land = ghgh[3]
# land2 = ghgh[4]
#
# output_250_hl = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/'
#                             'output_250_hl.csv', names=["No", "Address"])
#
# output_123_oa = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/'
#                             'output_123_oa.csv', names=["No", "Address"])
#
# s = land2['R']
#
# im = ghgh[0]['R']
#
# implot = plt.imshow(im, cmap=plt.cm.bone)
# metadata = ghgh[2]['R']['bbox']
# d = s[:, 1]
#
# s[:, 0] -= metadata[0]
#
# s[:, 1] -= metadata[1]
# center = im.shape[0]/2
# #out = flipping_amir(s, center)
#
#
# # put a red dot, size 40, at 2 locations:
# plt.scatter(x=s[:,0], y=s[:, 1], c='r', s=10)
#
# plt.show()



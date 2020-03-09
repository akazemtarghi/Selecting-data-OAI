from read_json import Read_json
import cv2
import matplotlib.pyplot as plt
import glob
import matplotlib.patches as patches
#
# prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/New folder (2)'
# dir_tibia = '/t.json'
# dir_femur = '/f.json'
def standard_ROI_amir(image,list_x_tibia,list_y_tibia):

    r = list_x_tibia[12] - list_x_tibia[9]
    box_midial = [list_x_tibia[9], list_y_tibia[9], list_x_tibia[11], list_y_tibia[11]+5]

    width = r//2
    y = box_midial[3]
    x = box_midial[0]+(r//4)
    roi1 = image[y: y + width, x: x + width]

    box_lateral = [list_x_tibia[5], list_y_tibia[5], list_x_tibia[6], list_y_tibia[6]+5]

    width = r//2
    y = box_lateral[3]
    x = box_lateral[0]+(r//4)

    roi2 = image[y: y + width, x: x + width]
    #
    # plt.figure()
    # plt.imshow(roi1)
    #
    # plt.figure()
    # plt.imshow(roi2)

    return roi1, roi2, r, box_midial, box_lateral


def show_patches(image, box1, box2, i, r):

    dir = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/examples of patches/'
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image, cmap=plt.cm.bone)
    # Create a Rectangle patch
    rect1 = patches.Rectangle((box1[0] + (r // 4), box1[3]), r // 2, r // 2, linewidth=1, edgecolor='r',
                              facecolor='none')
    rect2 = patches.Rectangle((box2[0] + (r // 4), box2[3]), r // 2, r // 2, linewidth=1, edgecolor='r',
                              facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    #plt.show()
    plt.savefig(dir + str(i) + '.jpg', bbox_inches='tight')


#
# #list_x, list_y = Read_json(dir_tibia)
# #list_x, list_y = Read_json(dir_femur)
#
# x_list = []
# y_list = []
# image_list = []
#
# for filename in glob.glob(prefix + '/*'):
#     a = filename + dir_tibia
#     # for tibia
#     list_x_tibia, list_y_tibia = Read_json(filename + dir_tibia)
#     # for femur
#     list_x_femur, list_y_femur = Read_json(filename + dir_femur)
#
#     x_list.append(list_x_tibia)
#     y_list.append(list_y_tibia)
#
#     for filename1 in glob.glob(filename + '/*.png'):
#         image = cv2.imread(filename1, 0)
#         image_list.append(image)
#
# I = image_list[1]
# x = x_list[0]
# y = y_list[0]
#
# roi1, roi2, r, box1, box2 = standard_ROI_amir(I, x, y)
#
#
#
# fig,ax = plt.subplots(1)
#
# # Display the image
# ax.imshow(image_list[1], cmap=plt.cm.bone)
# # Create a Rectangle patch
# rect1 = patches.Rectangle((box1[0]+(r//4), box1[3]+2), r//2, r//2, linewidth=1, edgecolor='r', facecolor='none')
# rect2 = patches.Rectangle((box2[0]+(r//4), box2[3]+2), r//2, r//2, linewidth=1, edgecolor='r', facecolor='none')
#
# # Add the patch to the Axes
# ax.add_patch(rect1)
# ax.add_patch(rect2)
#
# plt.show()
#
# plt.figure()
# plt.imshow(roi2, cmap=plt.cm.bone)
# plt.axis('off')
#
# plt.figure(figsize=(10.8, 10.8))
# plt.imshow(image_list[1], cmap=plt.cm.bone)
# plt.scatter(x=x_list[0], y=y_list[0], c='r', s=10)
# plt.axis('off')
#
#


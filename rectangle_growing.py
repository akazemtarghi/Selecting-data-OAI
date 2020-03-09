import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rec_growing(mask_femur, rotated_landmarks, side=None, name=None ):

    if name == 'femur':

        if side == 'lateral':

            x, y = rotated_landmarks[7, :]
            y = y - 5
            x = x + 5

            length_temp = 40
            width_temp = 30

            temp = mask_femur[y - width_temp:y, x - length_temp: x]

            while len(temp[temp == 0]) == 0:
                rec = temp
                width = width_temp
                length = length_temp

                width_temp = width_temp + 3
                length_temp = length_temp + 4
                temp = mask_femur[y - width_temp:y, x - length_temp: x]

            return rec, width-3, length-4, x, y

        elif side == 'medial':

            x, y = rotated_landmarks[9, :]
            y = y - 5


            length_temp = 40
            width_temp = 30

            temp = mask_femur[y - width_temp: y, x  : x + length_temp]

            while len(temp[temp == 0]) == 0:
                rec = temp
                width = width_temp
                length = length_temp

                width_temp = width_temp + 3
                length_temp = length_temp + 4
                temp = mask_femur[y - width_temp:y, x: x + length_temp]

            return rec, width-6, length-8, x, y

    elif name == 'tibia':

        if side == 'lateral':

            x = rotated_landmarks[7, 0]
            y = rotated_landmarks[6, 1] + 10

            length_temp = 40
            width_temp = 30

            temp = mask_femur[y: y + width_temp, x - length_temp:x ]

            while len(temp[temp == 0]) <= 10:
                rec = temp
                width = width_temp
                length = length_temp

                width_temp = width_temp + 3
                length_temp = length_temp + 4
                temp = mask_femur[y: y + width_temp, x - length_temp:x]

            return rec, width-3, length-4, x, y

        elif side == 'medial':

            x = rotated_landmarks[10, 0] - 10
            y = rotated_landmarks[11, 1] + 10

            length_temp = 40
            width_temp = 30

            plt.figure()
            plt.imshow(mask_femur)


            temp = mask_femur[y: y + width_temp, x : length_temp + x]



            while len(temp[temp == 0]) <= 10:

                rec = temp
                width = width_temp
                length = length_temp

                width_temp = width_temp + 3
                length_temp = length_temp + 4
                temp = mask_femur[y: y + width_temp, x : length_temp + x]

            return rec, width-6, length-8, x, y




def show_patches_femur(image,width, length, x, y,i, name=None):

    dir = 'C:/Users/Amir Kazemtarghi/Documents/data/images_for_annotations/examples of patches for tibia/'
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image, cmap=plt.cm.bone)
    # Create a Rectangle patch
    if name == 'lateral':
        rect1 = patches.Rectangle((x - length, y- width),
                                  length, width, linewidth=1,
                                  edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        # plt.show()
        plt.savefig(dir + str(i) + '_lat' + '.jpg', bbox_inches='tight')

    elif name == 'medial':

        rect1 = patches.Rectangle((x, y- width),
                                  length, width, linewidth=1,
                                  edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        # plt.show()
        plt.savefig(dir + str(i) + '_med' + '.jpg', bbox_inches='tight')




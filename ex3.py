import cv2
import numpy as np
import matplotlib.pyplot as plt
blur_filter = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
Num_of_levels = 5
BLUR_SIZE = 5


def construct_gaussian_pyramid(img):
    # Construct Gaussian pyramid for the input image
    # img - input image
    # Num_of_levels - number of levels in the pyramid
    # blur_filter - filter to use for blurring
    # return: Gaussian pyramid
    img_gaussian_pyramid = [img]
    for i in range(Num_of_levels - 1):
        # blur with the filter
        img = cv2.filter2D(img, -1, np.array(blur_filter).reshape(1, 5))
        img = img[::2, ::2]
        img_gaussian_pyramid.append(img)
    return img_gaussian_pyramid


def construct_laplacian_pyramid(img):
    # Construct Laplacian pyramid for the input image
    # img - input image
    # Num_of_levels - number of levels in the pyramid
    # blur_filter - filter to use for blurring
    # return: Laplacian pyramid
    img_gaussian_pyramid = construct_gaussian_pyramid(img)
    img_laplacian_pyramid = []
    for i in range(Num_of_levels - 1):
        # resize by 2 the small image
        image_help = cv2.resize(img_gaussian_pyramid[i + 1],
                                (img_gaussian_pyramid[i].shape[1], img_gaussian_pyramid[i].shape[0]))

        img_laplacian_pyramid.append(img_gaussian_pyramid[i] - image_help)


    img_laplacian_pyramid.append(
        img_gaussian_pyramid[Num_of_levels - 1])  # last level is the same as the last level of the Gaussian pyramid
    return img_laplacian_pyramid


def blending_two_images_seamlessly(img1, img2, mask):
    # img1 - michael image
    # img2 - arnold the terminator image
    # mask - mask image
    # Construct Laplacian pyramid for both images
    # Construct Gaussian pyramid for the mask
    guassian_pyramid_img1 = construct_gaussian_pyramid(img1)
    plt.figure()
    for i in range(Num_of_levels):
        plt.subplot(1, Num_of_levels, i + 1)
        plt.imshow(guassian_pyramid_img1[i], cmap='gray')
        plt.title('Level ' + str(i))
    guassian_pyramid_img2 = construct_gaussian_pyramid(img2)
    plt.figure()
    for i in range(Num_of_levels):
        plt.subplot(1, Num_of_levels, i + 1)
        plt.imshow(guassian_pyramid_img2[i], cmap='gray')
        plt.title('Level ' + str(i))

    Laplacian_pyramid_img1 = construct_laplacian_pyramid(img1)
    #plot all the levels of the Laplacian pyramid next to each other
    plt.figure()
    for i in range(Num_of_levels):
        plt.subplot(1, Num_of_levels, i + 1)
        plt.imshow(Laplacian_pyramid_img1[i], cmap='gray')
        plt.title('Level ' + str(i))
    plt.show()


    Laplacian_pyramid_img2 = construct_laplacian_pyramid(img2)
    #plot all the levels of the Laplacian pyramid next to each other
    plt.figure()
    for i in range(Num_of_levels):
        plt.subplot(1, Num_of_levels, i + 1)
        plt.imshow(Laplacian_pyramid_img2[i], cmap='gray')
        plt.title('Level ' + str(i))
    plt.show()
    Gaussian_pyramid_mask = construct_gaussian_pyramid(mask)

    Laplacian_pyramid_blend = []
    for k in range(Num_of_levels):
        Laplacian_pyramid_blend.append(
            (Gaussian_pyramid_mask[k] * Laplacian_pyramid_img1[k]) + (1 - Gaussian_pyramid_mask[k]) *
            Laplacian_pyramid_img2[k])


    #plot all the levels of the Laplacian pyramid blend next to each other

    plt.figure()
    for i in range(Num_of_levels):
        plt.subplot(1, Num_of_levels, i + 1)
        plt.imshow(Laplacian_pyramid_blend[i], cmap='gray')
        plt.title('Level ' + str(i))
    plt.show()

    # sum all the levels of the Laplacian pyramid to get the blended image
    # we want the blended image to be in the size of the original image , so it should be in the size of the first laplacian pyramid
    img_blended = Laplacian_pyramid_blend[-1]
    for i in range(Num_of_levels - 2, -1, -1):
        img_blended = cv2.resize(img_blended,
                                 (Laplacian_pyramid_blend[i].shape[1], Laplacian_pyramid_blend[i].shape[0]))
        img_blended = img_blended + Laplacian_pyramid_blend[i]
        #normalize the image
        img_blended_min = np.min(img_blended)
        img_blended_max = np.max(img_blended)
        img_blended = (img_blended - img_blended_min) / (img_blended_max)



    return img_blended

def hybrid_images(img1,img2):
    # we should take the low frequencies of the first image and the high frequencies of the second image
    # we will do it by laplacian pyramid
    lap_img1=construct_laplacian_pyramid(img1)
    lap_img2=construct_laplacian_pyramid(img2)
    #we want to take the low frequencies of the first image and then the high frequencies of the second image
    hybrid_img_lap=[]
    for i in range(Num_of_levels):
        #until level 2 we take the low frequencies of the first image
        if i<3:
            hybrid_img_lap.append(lap_img1[i])
        #from level 2 we take the high frequencies of the second image
        else:
            hybrid_img_lap.append(lap_img2[i])

    #sum all the levels of the Laplacian pyramid to get the blended image
    hybrid_image=hybrid_img_lap[-1]
    for i in range(Num_of_levels-2,-1,-1):
        hybrid_image=cv2.resize(hybrid_image,(hybrid_img_lap[i].shape[1],hybrid_img_lap[i].shape[0]))
        hybrid_image=hybrid_image+hybrid_img_lap[i]
        #normalize the image
        hybrid_image_min=np.min(hybrid_image)
        hybrid_image_max=np.max(hybrid_image)
        hybrid_image=(hybrid_image-hybrid_image_min)/(hybrid_image_max)
    return hybrid_image


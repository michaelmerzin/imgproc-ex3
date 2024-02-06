import cv2

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
        img = cv2.blur(img, (BLUR_SIZE, BLUR_SIZE))
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
        # resize the small image to the size of the bigger image
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

    Laplacian_pyramid_img1 = construct_laplacian_pyramid(img1)
    Laplacian_pyramid_img2 = construct_laplacian_pyramid(img2)
    Gaussian_pyramid_mask = construct_gaussian_pyramid(mask)
    # display the mask pyramid


    # Create a third Laplacian Pyramid Lc using the following formula:
    # Lc(i,j) = G(i,j) * La(i,j) + (1 - G(i,j)) * Lb(i,j)
    # G is the Gaussian pyramid of the mask
    # La and Lb are Laplacian pyramids of the two input images
    # Lc is the Laplacian pyramid of the blended image
    # i, j are the pixel coordinates
    # make it for each level k of the pyramid
    Laplacian_pyramid_img3 = []
    for k in range(Num_of_levels):
        Laplacian_pyramid_img3.append(
            (Gaussian_pyramid_mask[k] * Laplacian_pyramid_img1[k]) + (1 - Gaussian_pyramid_mask[k]) *
            Laplacian_pyramid_img2[k])

    # sum all the levels of the Laplacian pyramid to get the blended image
    # we want the blended image to be in the size of the original image , so it should be in the size of the first laplacian pyramid
    img_blended = Laplacian_pyramid_img3[Num_of_levels - 1]
    for i in range(Num_of_levels - 2, -1, -1):
        img_blended = cv2.resize(img_blended, (Laplacian_pyramid_img3[i].shape[1], Laplacian_pyramid_img3[i].shape[0]))
        img_blended = img_blended + Laplacian_pyramid_img3[i]

    return img_blended

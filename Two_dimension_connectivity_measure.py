# Python program to demonstrate erosion and
# dilation of images.
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def circular_kernel(radius):
    """
    Returns a perfect circular structuring element.
    radius: circle radius in pixels
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    k = mask.astype(np.uint8)
    return k

kernel_size=7
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
kernel=circular_kernel(radius=kernel_size//2)
lamda=0.5

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper

#@measure_time
def connectivity_analysis(binary_image):
    erosion_number = 0
    data_list = []

    # load image
    img = binary_image
    #img = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
    #img=255-img
    #cv2.imshow('original', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #gray_mask = (img > 50) & (img < 200)
    #img = np.where(gray_mask, 0, 255).astype(np.uint8)
    #img = cv2.bitwise_not(img)
    #shape_area = np.sum(img == 255)
    #cv2.imshow('original', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # size of kernal is propotional to the degree of erosion and dilation.
    #kernel = np.ones((kernel_size, kernel_size), np.uint8)

    num_labels, labels_im = cv2.connectedComponents(img,connectivity=8)
    original_component_number = num_labels - 1

    while True:
        img_erosion = cv2.erode(img, kernel, iterations=erosion_number)
        #cv2.imshow('erosion', img_erosion)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        num_labels, labels_im = cv2.connectedComponents(img_erosion,connectivity=8)
        num_connected_components = num_labels - 1
        if original_component_number > 1:
            if num_connected_components == 1:
                #print("shape after erosion is connected, dilation number is 0")
                data_list.append(0)
                erosion_number += 1
            elif num_connected_components == 0:
                #print("shape vanished, dilation number if inf,function closed")
                break
            else:
                #print("original shape is not connected, analyze partial connectivity.")
                current_image = img_erosion
                current_components_number = num_connected_components
                dilation_number = 0
                while current_components_number > 1:
                    img_dilation = cv2.dilate(current_image, kernel, iterations=1)
                    current_image = img_dilation
                    num_labels, labels_im = cv2.connectedComponents(current_image,connectivity=8)
                    current_components_number = num_labels - 1
                    dilation_number += 1

                    # can print result of dilation for each iteration
                    # cv2.imshow('intersection', current_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                #print(f'After {dilation_number} times of dilation, the shape become connected')
                #print(f'data tuple to be record:{erosion_number},{dilation_number}')
                data_list.append(dilation_number)
                erosion_number += 1
        else:
            if num_connected_components == 1:
                #print("shape is still connected, dilation number is 0")
                data_list.append(0)
                erosion_number += 1
            elif num_connected_components == 0:
                #print("shape vanished, dilation number if inf,function closed")
                break
            else:
                #print(f"shape is separated, Number of connected components: {num_connected_components}")
                # Create a blank image to display components
                output_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                # Assign colors to each component
                colors = []
                for i in range(1, num_labels):
                    colors.append(np.random.randint(0, 255, size=3))  # random color to show components
                    # Apply the colors to the output image
                for label in range(1, num_labels):
                    output_image[labels_im == label] = colors[label - 1]
                # display components

                #cv2.imshow('Connected Components', output_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()


                current_image = img_erosion
                current_components_number = num_connected_components
                dilation_number = 0
                while current_components_number > 1:
                    img_dilation = cv2.dilate(current_image, kernel, iterations=1)
                    img_stepwise_intersection = cv2.bitwise_and(img_dilation, img)
                    current_image = img_stepwise_intersection
                    num_labels, labels_im = cv2.connectedComponents(current_image,connectivity=8)
                    current_components_number = num_labels - 1
                    dilation_number += 1

                    # can print result of dilation for each iteration
                    # cv2.imshow('intersection', current_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                #print(f'After {dilation_number} times of dilation, the shape restored to connected')
                #print(f'data tuple to be record:{erosion_number},{dilation_number}')
                data_list.append(dilation_number)
                erosion_number += 1

                # cv2.imshow('Input', img)
                # cv2.imshow('Erosion', img_erosion)
                # cv2.imshow('Image after dilation', current_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    if len(data_list) != 1:
        #print(data_list)
        y_values = np.exp(-lamda * np.array(data_list))
        #print(y_values)
        integral = np.trapz(y_values)
        connected_connectivity = integral / (len(data_list) - 1)
        #general_connectivity = shape_area*connected_connectivity
        #print(f"The area of the shape in the original image is {shape_area} pixels.")
        #print(f"the connected connectivity of given shape is {connected_connectivity}")
        #print(f"the general connectivity of given shape is {general_connectivity}")
        return connected_connectivity
    return 0



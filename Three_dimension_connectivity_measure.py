import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import ball, binary_erosion, binary_dilation
from skimage.measure import label
import random
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


# Function to handle key events
def on_key(event):
    if event.key == 'q':  # Press 'q' to close the figure (like waitKey behavior)
        plt.close(event.canvas.figure)


def random_color():
    return [random.random() for _ in range(3)]



def multiple_erosions(original_object, kernel, erosion_number):
    eroded_object = original_object
    for _ in range(erosion_number):
        eroded_object = binary_erosion(eroded_object, footprint=kernel)
    return eroded_object



def plot_3d_object(labeled_object, num_components, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for component in range(1, num_components + 1):
        # Get the coordinates of the current component
        x, y, z = np.nonzero(labeled_object == component)

        # Plot the current component with a random color
        ax.scatter(x, y, z, color=random_color(), s=20, marker='o', label=f'Component {component}')
    ax.set_xlim([0, labeled_object.shape[0]])
    ax.set_ylim([0, labeled_object.shape[1]])
    ax.set_zlim([0, labeled_object.shape[2]])
    ax.set_title(title)
    ax.set_axis_off()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


#@measure_time
def connectivity_analysis(object_matrix, kernel_size, lamda):
    erosion_number = 0
    data_list = []

    #radius = 20  # Radius of the sphere
    original_object = object_matrix
    #labeled_object, num_components = label(original_object, connectivity=2, return_num=True)
    #plot_3d_object(labeled_object, num_components, 'Fault-Tolerant Workspace')
    #cv2.imshow('original', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # gray_mask = (img > 50) & (img < 200)
    # img = np.where(gray_mask, 0, 255).astype(np.uint8)
    # img = cv2.bitwise_not(img)
    #object_volume = np.sum(original_object)
    #print(object_volume)
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # size of kernal is propotional to the degree of erosion and dilation.
    kernel = ball(kernel_size)

    labeled_object, num_components = label(original_object, connectivity=2, return_num=True)
    original_component_number = num_components
    # plot_3d_object(labeled_object, num_components, 'Original Object')

    while True:
        eroded_object = multiple_erosions(original_object, kernel, erosion_number)
        labeled_object, num_components = label(eroded_object, connectivity=2, return_num=True)
        num_connected_components = num_components
        # cv2.imshow('erosion', img_erosion)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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
                current_object = eroded_object
                current_components_number = num_connected_components
                dilation_number = 0
                while current_components_number > 1:
                    dilated_object = binary_dilation(current_object, footprint=kernel)
                    current_object = dilated_object
                    labeled_object, num_components = label(current_object, connectivity=2, return_num=True)
                    #plot_3d_object(labeled_object, num_components, 'Fault-Tolerant Workspace')
                    current_components_number = num_components
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
                """
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for component in range(1, num_components + 1):
                    # Get the coordinates of the current component
                    x, y, z = np.nonzero(labeled_object == component)

                    # Plot the current component with a random color
                    ax.scatter(x, y, z, color=random_color(), s=100, marker='o', label=f'Component {component}')
                ax.set_xlim([-50, 50])
                ax.set_ylim([-50, 50])
                ax.set_zlim([-50, 50])

                ax.set_axis_off()
                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()
                """

                current_object = eroded_object
                current_components_number = num_connected_components
                dilation_number = 0
                while current_components_number > 1:
                    dilated_object = binary_dilation(current_object, footprint=kernel)
                    object_stepwise_intersection = np.logical_and(dilated_object,original_object)
                    current_object = object_stepwise_intersection
                    labeled_object, num_components = label(current_object, connectivity=2, return_num=True)
                    current_components_number = num_components
                    dilation_number += 1

                    # can print result of dilation for each iteration
                    # cv2.imshow('intersection', current_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                # print(f'After {dilation_number} times of dilation, the shape become connected')
                # print(f'data tuple to be record:{erosion_number},{dilation_number}')
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
        integral = np.trapz(y_values)
        connected_connectivity = integral / (len(data_list) - 1)
        #general_connectivity = object_volume*connected_connectivity
        #print(f"The volume of the shape in the original image is {object_volume} pixels.")
        #print(f"the connected connectivity of given shape is {connected_connectivity}")
        #print(f"the general connectivity of given shape is {general_connectivity}")
        return connected_connectivity
    return 0




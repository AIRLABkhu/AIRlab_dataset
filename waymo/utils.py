
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils

'''
2022.08.16 김대호
Waymo Open Dataset tutorial 참조 
'''


def distortion_correction(frame):
    w = frame.context.camera_calibrations[0].width
    h =  frame.context.camera_calibrations[0].height
    f_x, f_y, c_u, c_v, k_1, k_2, p_1, p_2, k_3= frame.context.camera_calibrations[0].intrinsic 

    mtx = np.array([[f_x, 0, c_u],[0, f_y, c_v],[0, 0, 1]])
    dist = np.array([[k_1, k_2, p_1, p_2, k_3]])
    print('1-1. Calibration: intrinsic matrix')
    print (mtx)
    print('1-2. Calibration: distortion coefficients')
    print (dist)

    plt.figure(figsize=(18, 10))

    frame_jpg = tf.image.decode_jpeg(frame.images[1].image)

    plt.subplot(1, 2, 1)
    plt.imshow (frame_jpg)
    plt.title('distortion')
    plt.axis = ('off')

    frame_jpg = np.array(frame_jpg)
    undist, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img_undist = cv2.undistort(frame_jpg, mtx, dist, None, undist)

    plt.subplot(1, 2, 2)
    plt.imshow(img_undist)
    plt.title('undistortion')
    plt.axis = ('off')

def origianl_images_to_panorama(frames_unordered):
    camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                                open_dataset.CameraName.FRONT_LEFT,
                                open_dataset.CameraName.FRONT,  
                                open_dataset.CameraName.FRONT_RIGHT,
                                open_dataset.CameraName.SIDE_RIGHT] 
                              
    frames_ordered = []

    for frame in frames_unordered:
        image_proto_dict = {image.name : image.image for image in frame.images}
        frames_ordered.append([image_proto_dict[name] for name in camera_left_to_right_order])

        def _pad_to_common_shape(image):
            return np.pad(image, [[1280 - image.shape[0], 0], [0, 0], [0, 0]])

    images_decode = [[tf.image.decode_jpeg(frame) for frame in frames ] for frames in frames_ordered]
    padding_images = [[_pad_to_common_shape(image) for image in images ] for images in images_decode]
    panorama_image_no_concat = [np.concatenate(image, axis=1) for image in padding_images]
    panorama_image = np.concatenate(panorama_image_no_concat, axis=0)

    plt.figure(figsize=(64, 60))
    plt.imshow(panorama_image)
    plt.grid(False)
    plt.axis('off')
    plt.show()
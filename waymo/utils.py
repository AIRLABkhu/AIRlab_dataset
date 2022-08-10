
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf

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
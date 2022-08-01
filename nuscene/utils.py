import cv2
import numpy as np
import natsort
import glob
import os

def make_panorama(sample, index):
    img_list = []
    for frame in sample:
        # if frame != None:
        img = cv2.imread(os.path.join('../nuimage_sample', frame['filename']))
        # 900 * 1600 to 255 * 400 (/4)
        img = cv2.resize(img, dsize=(400,255))
        img_list.append(img)

    img_concat = np.hstack((np.asarray(i)) for i in img_list)
    cv2.imwrite(f'../pano/{index}.png', img_concat)

    return img_concat

def make_video(img_path, dst_path, fps):
    fps = fps
    frame_array = []

    for file_path in natsort.natsorted(glob.glob(img_path)):
        img = cv2.imread(file_path)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(dst_path, fourcc, fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        video.write(frame_array[i])

    video.release()

# calculate the field of view of the camera
def cal_fov(img_w, img_h, intrinsic):
    fov_x = np.rad2deg(2 * np.arctan2(img_w, 2 * intrinsic[0][0]))
    fov_y = np.rad2deg(2 * np.arctan2(img_h, 2 * intrinsic[1][1]))
    return fov_x, fov_y

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def line_plane_cross_point(line, plane):
    a, b, c, d = line @ plane
    x = a/d
    y = d/d
    z = d/d
    return x, y, z

def line_intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
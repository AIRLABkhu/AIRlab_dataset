import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils
from utils import distortion_correction, origianl_images_to_panorama

FILE_NAME = 'set file path'

if __name__ == '__main__' : 
    dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')

    frames_unordered = []
    frames_with_seg = []

    sequence_id = None

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if frame.images[0].camera_segmentation_label.panoptic_label: # frame.images[i] i = 0..4  
            frames_with_seg.append(frame)
            frames_unordered.append(frame)
            if sequence_id is None:
                sequence_id = frame.images[0].camera_segmentation_label.sequence_id
                print(sequence_id)
            if frame.images[0].camera_segmentation_label.sequence_id != sequence_id or len(frames_with_seg) > 0:
                break
    
    
    distortion_correction(frame)
    origianl_images_to_panorama(frames_unordered)
from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)
        #SCARED Dataset
        """self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""
                
        #256 / 320
        #fx769.807403688120 fy769.720558534159 cx675.226397736271 cy548.903474592445 k1-0.454260397098776 k20.179156666748519 k3-0.0285017743214105 p1-0.00134889190333418 p20.000738912923806121 skew-0.141152521412316
        """self.K = np.array([[2.40, -0.141152521412316, 2.11, 0],
                           [0, 3.00,2.14, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""
        #RNNSLAM synthetic dataset
        #256 / 320
        #fx = 155.0029785575048732943469785575 fy= 165.33651046133853151397011046134 cx = 128.0; cy = 160.0;
        self.K = np.array([[0.48, 0, 0.4, 0],
                           [0, 0.64,0.62, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        #Colon10k dataset
        #256 / 320
        #Camera Intrinsics: Pinhole fx=145.4410 fy=145.4410 cx=135.6993 cy=107.8946 width=270 height=216
        """self.K = np.array([[0.4545, 0, 0.4240, 0],
                           [0, 0.5681,0.4214, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        #SCATER
        #f_str = "{}{}".format(frame_index, self.img_ext)
        #image_path = os.path.join(self.data_path, folder, "data", f_str)
        #COLON10k
        f_str=str(frame_index) + self.img_ext
        image_path = os.path.join(self.data_path, folder, f_str)
        #RNNSLAM
            
        return image_path

class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        #SCATER
        #f_str = "{}{}".format(frame_index, self.img_ext)
        #image_path = os.path.join(self.data_path, folder, "data", f_str)
        #COLON10k
        f_str=str(frame_index) + self.img_ext
        image_path = os.path.join(self.data_path, folder, f_str)
            
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt



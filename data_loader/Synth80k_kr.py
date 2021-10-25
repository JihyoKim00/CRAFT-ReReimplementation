import sys
import os

import torch
import torch.utils.data as data
import scipy.io as scio
import re
import itertools

import random
from PIL import Image
import torchvision.transforms as transforms
import Polygon as plg
import numpy as np
import cv2

try:
    from data_loader.craft_base_dataset import craft_base_dataset
except:
    from craft_base_dataset import craft_base_dataset


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.file_utils import *
from util.gaussian import GaussianTransformer
from util.watershed import watershed
from util.mep import mep
import util.imgproc as imgproc
import h5py







class Synth80k_kr(craft_base_dataset):

    def __init__(self, synthtext_folder, target_size=768, viz=False, debug=False, perform_input_data_corruption=False):
        super(Synth80k_kr, self).__init__(target_size, viz, debug, perform_input_data_corruption=perform_input_data_corruption)
        self.synthtext_folder = synthtext_folder
        self.gt = self.check_data(synthtext_folder)
        self.gt_name = list(self.gt['data'].keys())


    def check_data(self, path):

        folder, ext = os.path.splitext(path)
        if ext == '.h5':
            gt = h5py.File(path, 'r')
        else:
            gt = h5py.File(os.path.join(path, 'dset_kr.h5'), 'r')

        return gt



    def __len__(self):
        return len(self.gt['data'].keys())


    def get_imagename(self, index):
        return self.gt_name[index][0]

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''

        image = self.gt['data'][self.gt_name[index]][...]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        charBB = self.gt['data'][self.gt_name[index]].attrs['charBB']
        #wordBB = self.gt['data'][self.gt_name[index]].attrs['wordBB']
        txt = self.gt['data'][self.gt_name[index]].attrs['txt']
        imgpath = self.gt['data'][self.gt_name[index]].attrs['imgpath']

        _charbox = charBB.transpose((2, 1, 0))


        try:
            words = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
        except:

            txt = [t.decode('UTF-8') for t in txt]
            words = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]



        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)


        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, imgpath


if __name__ == '__main__':


    path = '/data/workspace/woans0104/SynthText_kr-master/data/background/gen'
    dataloader = Synth80k_kr(path,viz=False, target_size=768, perform_input_data_corruption=True)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (opimage, region_scores, affinity_scores, confidence_mask, confidences_mean, unnormalized_images, img_paths) in enumerate(train_loader):
        total += 1
        print('$'*100)




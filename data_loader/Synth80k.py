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


class Synth80k(craft_base_dataset):

    def __init__(self, synthtext_folder, target_size=768, viz=False, debug=False, perform_input_data_corruption=True):
        super(Synth80k, self).__init__(target_size, viz, debug, perform_input_data_corruption=perform_input_data_corruption)
        self.synthtext_folder = synthtext_folder
        gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))

        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''



        img_path = os.path.join(self.synthtext_folder, self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _charbox = self.charbox[index].transpose((2, 1, 0))
        #image = random_scale(image, _charbox, self.target_size)

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
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


        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, img_path


if __name__ == '__main__':

    dataloader = Synth80k('/home/data/ocr/detection/SynthText/SynthText',viz=True, target_size=768, perform_input_data_corruption=False)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (opimage, region_scores, affinity_scores, confidence_mask, confidences_mean, unnormalized_images, img_paths) in enumerate(train_loader):
        total += 1
        print(opimage.shape)


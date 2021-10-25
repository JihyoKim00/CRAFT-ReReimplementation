import torch
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import logging
import math
import json
from craft_base_dataset import craft_base_dataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.file_utils import *

class AI_HUB(craft_base_dataset):
    """
    dataset formats:
    * all image files end with ".JPG"

    Directory structure:
    textinthewild_data
    ├── all_image
    |   ├── image1.jpg
    |   ├── image2.jpg
    |   ├── image3.jpg
    ├── book_crop
    ├── sign_crop
    ├── product_crop



    """
    def __init__(self, path, target_size=768, viz=False, debug=False, data_corruption=False):
        super(AI_HUB, self).__init__(target_size, viz, debug, data_corruption)
        self.img_paths = []
        self.xml_paths = '/home/data/ocr/ai-hub-data/textinthewild_data/textinthewild_data_info.json'
        self.xml_data = ''
        self.search_data = {}
        self.target_size = target_size
        self.debug = debug
        self.skipped_images = 0


        with open(self.xml_paths, 'r', encoding='utf-8') as f:
            self.xml_data = json.load(f)

        for i in range(len(self.xml_data['images'])):
            for k, v in self.xml_data['images'][i].items():
                self.search_data.update({v: k})


        for lang_dir in os.listdir(path):
            if 'crop' not in lang_dir and not lang_dir.endswith('zip') and not lang_dir.endswith('json'):
                lang_path = os.path.join(path, lang_dir)

                for file in os.listdir(lang_path):
                    img_file = os.path.join(lang_path, file)

                    is_valid = self.is_valid(file,self.search_data)

                    if is_valid:
                        self.img_paths.append(img_file)
                    else:
                        self.skipped_images += 1

    def __len__(self):
        return len(self.img_paths)


    def is_valid(self, img_filename, xml_data):
        if xml_data.get(img_filename) != None:
            return True
        else:
            return False



    def read_xml(self, xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        resolution = root[0][1].attrib
        img_width = resolution["x"]
        img_height = resolution["y"]

        words = root[0][2]
        words_text = []
        char_bboxes = []
        for word in words:
            word_text = ""
            char_bbox = np.ndarray((len(word), 4, 2), np.float32)
            for char_idx, char in enumerate(word):
                char = char.attrib

                # upper-left corner, height, and width (from the corner)
                char_x = int(char["x"])
                char_y = int(char["y"])
                char_width = int(char["width"])
                char_height = int(char["height"])

                if math.isnan(char_x) or math.isnan(char_y) or math.isnan(char_width) or math.isnan(char_height):
                    raise ValueError("Corrupt data in file: " + xml_path +"\n Nan in one of the coordinates")

                # upper-left corner
                char_bbox[char_idx][0][0] = char_x
                char_bbox[char_idx][0][1] = char_y

                # upper-right corner
                char_bbox[char_idx][1][0] = char_x + char_width
                char_bbox[char_idx][1][1] = char_y

                # lower-right corner
                char_bbox[char_idx][2][0] = char_x + char_width
                char_bbox[char_idx][2][1] = char_y + char_height

                # lower-left corner
                char_bbox[char_idx][3][0] = char_x
                char_bbox[char_idx][3][1] = char_y + char_height

                word_text += char["char"]

            if len(word_text) != 0:
                words_text.append(word_text)
            if len(char_bbox) != 0:
                char_bboxes.append(char_bbox)

        return char_bboxes, words_text, (img_width, img_height)



    def read_xml(self, xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        resolution = root[0][1].attrib
        img_width = resolution["x"]
        img_height = resolution["y"]

        words = root[0][2]
        words_text = []
        char_bboxes = []
        for word in words:
            word_text = ""
            char_bbox = np.ndarray((len(word), 4, 2), np.float32)
            for char_idx, char in enumerate(word):
                char = char.attrib

                # upper-left corner, height, and width (from the corner)
                char_x = int(char["x"])
                char_y = int(char["y"])
                char_width = int(char["width"])
                char_height = int(char["height"])

                if math.isnan(char_x) or math.isnan(char_y) or math.isnan(char_width) or math.isnan(char_height):
                    raise ValueError("Corrupt data in file: " + xml_path +"\n Nan in one of the coordinates")

                # upper-left corner
                char_bbox[char_idx][0][0] = char_x
                char_bbox[char_idx][0][1] = char_y

                # upper-right corner
                char_bbox[char_idx][1][0] = char_x + char_width
                char_bbox[char_idx][1][1] = char_y

                # lower-right corner
                char_bbox[char_idx][2][0] = char_x + char_width
                char_bbox[char_idx][2][1] = char_y + char_height

                # lower-left corner
                char_bbox[char_idx][3][0] = char_x
                char_bbox[char_idx][3][1] = char_y + char_height

                word_text += char["char"]

            if len(word_text) != 0:
                words_text.append(word_text)
            if len(char_bbox) != 0:
                char_bboxes.append(char_bbox)

        return char_bboxes, words_text, (img_width, img_height)

    def read_img(self, img_path, char_bboxes):
        """Read image from disk. When read, by default, the bytes are arranged in BGR, not RGB.
        So we re-order the colors to RGB with cv2.cvtColor.

        source:https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor

        :param img_path:
        :param char_bboxes:
        :return:
        """

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_imagename(self, index):
        return self.img_paths[index].split("/")[-1]

    def load_image_gt_and_confidencemask(self, index):
        if index < 0 or index >= len(self):
            raise ValueError("index out of range")

        img_path = self.img_paths[index]
        xml_data = self.xml_data

        char_bboxes, words , _= self.read_xml(xml_path)
        img = self.read_img(img_path, char_bboxes)

        # since KAIST has character-level bboxes, confidence is 100 for all char bboxes.
        confidence = np.ones((len(words)), np.float32)
        confidence_mask = np.ones((img.shape[0], img.shape[1]), np.float32)

        return img, char_bboxes, words, confidence_mask, confidence, img_path

if __name__ == '__main__':

    dataloader = AI_HUB('/home/data/ocr/ai-hub-data/textinthewild_data',viz=True, target_size=768, data_corruption=False)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (opimage, region_scores, affinity_scores, confidence_mask, confidences_mean, unnormalized_images, img_paths) in enumerate(train_loader):
        total += 1
        print(total)
        import ipdb;ipdb.set_trace()


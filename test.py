"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from util import craft_utils, imgproc, file_utils
import logging

from model.craft import CRAFT

from collections import OrderedDict

from tqdm import tqdm
from util import writer as craft_writer
from util.mseloss import Maploss
from eval.script import eval_2015


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(args, net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 2240,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass

    y, _ = net(x)

    #out1 = y[:, :, :, 0].copy().cuda()
    #out2 = y[:, :, :, 1].copy().cuda()


    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time :

    logging.info("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def test(args, ckpt_path):
    # load net

    net = CRAFT()
    logging.info('Loading weights from checkpoint {}'.format(ckpt_path))

    net_param = torch.load(ckpt_path)
    net.load_state_dict(copyStateDict(net_param['craft']))

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        logging.info("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold,
                                             args.low_text, True, args.poly)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    logging.info("elapsed time : {}s".format(time.time() - t))






def post_processing(image_path, score_text, score_link, text_threshold, link_threshold, low_text, poly):


    image = imgproc.loadImage(image_path)



    score_text = score_text.clone().detach().cpu().data.numpy()[0,:,:]
    score_link = score_link.clone().detach().cpu().data.numpy()[0,:,:]


    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 768,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio


    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)


    return image, boxes, polys, ret_score_text



def save_pic(image_path, score_text, image, polys, result_folder, gt_file_path, crop=False, gt_bbox=False):

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)


    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)


    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder,gt_bbox=gt_file_path,crop=crop)



# TO-do
def validation(args, net, loader, logger, epoch, poly, viz, result_folder):
    losses = craft_writer.AverageMeter()
    net.eval()
    t = time.time()

    print('start validation')
    # import ipdb;ipdb.set_trace()
    with torch.no_grad():
        for idx,(images, gh_label, gah_label, mask, _, unnormalized_images, img_paths) in enumerate(loader):

            images = images.cuda()
            gh_label = gh_label.cuda()
            gah_label = gah_label.cuda()
            mask = mask.cuda()

            out, _ = net(images)

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()

            criterion = Maploss()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            # log update
            losses.update(loss.item(), 1)

            if viz:
                image, boxes, polys, ret_score_text = post_processing(img_paths[0], out1, out2,
                                                                      args.text_threshold, args.link_threshold,
                                                                      args.low_text, poly)

                save_pic(img_paths[0], ret_score_text, image, polys, gt_file_path=args.gt_file_path, result_folder=result_folder,
                         crop=False)

    #resDict = eval_2015(result_folder)

    logger.write([epoch, losses.avg])
    print(f'Validation Loss: {losses.avg:0.5f} ')
    print('#----------------------------------------------------#')
    return losses.avg


# #TO-do
# def validation(net, loader, logger, epoch):
#
#     losses = craft_writer.AverageMeter()
#     net.eval()
#     t = time.time()
#
#     print('start validation')
#     #import ipdb;ipdb.set_trace()
#     with torch.no_grad():
#         for images, gh_label, gah_label, mask, _, unnormalized_images, img_paths in loader:
#
#             images = images.cuda()
#             gh_label = gh_label.cuda()
#             gah_label = gah_label.cuda()
#             mask = mask.cuda()
#
#             out, _ = net(images)
#
#
#             out1 = out[:, :, :, 0].cuda()
#             out2 = out[:, :, :, 1].cuda()
#
#             criterion = Maploss()
#             loss = criterion(gh_label, gah_label, out1, out2, mask)
#
#             # log update
#             losses.update(loss.item(), 1)
#
#
#     logger.write([epoch, losses.avg])
#     print(f'Validation Loss: {losses.avg:0.5f} ')
#     print('#----------------------------------------------------#')
#     return losses.avg
#
#




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')

    args = parser.parse_args()

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files('/data/CRAFT-pytorch/test')

    result_folder = '/data/CRAFT-pytorch/result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
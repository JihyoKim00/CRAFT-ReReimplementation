from test import test_net
import time
import os
from util import imgproc, file_utils
import cv2
from eval.script import eval_2015
import torch
import torch.backends.cudnn as cudnn
import cv2

from util import craft_utils, imgproc, file_utils
import logging
from model.craft import CRAFT

from test import copyStateDict

#from eval.script import eval_2013


# def eval2013(craft, test_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
#     image_list, _, _ = file_utils.get_files(test_folder)
#     t = time.time()
#     res_gt_folder = os.path.join(result_folder, 'gt')
#     res_mask_folder = os.path.join(result_folder, 'mask')
#     # load data
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
#         image = imgproc.loadImage(image_path)
#
#         bboxes, polys, score_text = test_net(craft, image, text_threshold, link_threshold, low_text, True, False, 980,
#                                              1.5, False)
#
#         # save score text
#         filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
#         cv2.imwrite(mask_file, score_text)
#
#         file_utils.saveResult13(image_path, polys, dirname=res_gt_folder)
#
#     eval_2013(res_gt_folder)
#     print("elapsed time : {}s".format(time.time() - t))


def eval2015(craft, test_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):


    net = CRAFT(use_vgg16_pretrained=False)
    logging.info('Loading weights from checkpoint {}'.format(craft))

    net_param = torch.load(craft)



    try:
        net.load_state_dict(copyStateDict(net_param['craft']))
    except:
        net.load_state_dict(copyStateDict(net_param))

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)


    image_list, _, _ = file_utils.get_files(test_folder)
    t = time.time()
    if not os.path.exists(os.path.join(result_folder, 'gt')) and not os.path.exists(os.path.join(result_folder, 'mask')):
        os.makedirs(os.path.join(result_folder, 'gt'))
        os.makedirs(os.path.join(result_folder, 'mask'))

    res_gt_folder = os.path.join(result_folder, 'gt')
    res_mask_folder = os.path.join(result_folder, 'mask')


    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(args, net, image, text_threshold, link_threshold, low_text, True, False)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image, bboxes, dirname=res_gt_folder)




    #eval_2015(os.path.join(result_folder, 'gt'))
    print("elapsed time : {}s".format(time.time() - t))







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--craft', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--test_folder', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--result_folder', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')

    args = parser.parse_args()


    eval2015(craft=args.craft, test_folder=args.test_folder,result_folder=args.result_folder,
             text_threshold=args.text_threshold, link_threshold=args.link_threshold, low_text=args.low_text)
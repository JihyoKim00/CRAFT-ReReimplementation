"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from util import file_utils, imgproc
from util import writer

import os
import torch

""" auxilary functions """
# unwarp coorodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        mask = labels==k
        mask = textmap[mask]
        if np.max(mask) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

"""
def save_net_outputs(text_heat_map, link_heat_map, image_paths, images=None, output_path="net_results", text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                     canvas_size=2240, mag_ratio=2.0, poly=False):
    batch_size = text_heat_map.shape[0]
    assert batch_size == len(image_paths), "number of image paths doesn't match the bathc size of network output"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for batch in range(batch_size):
        image_path = image_paths[batch]

        if images is None:
            image = imgproc.loadImage(image_path)
        else:
            image = images[batch].cpu().data.numpy()

        _, target_ratio, _ = imgproc.resize_aspect_ratio(image, canvas_size,
                                                          interpolation=cv2.INTER_LINEAR,
                                                          mag_ratio=mag_ratio)

        ratio_h = ratio_w = 1 / target_ratio

        # make score and link map
        score_text = text_heat_map[batch].cpu().data.numpy()
        score_link = link_heat_map[batch].cpu().data.numpy()

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        score_text = imgproc.cvt2HeatmapImg(render_img)

        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        masked_img_path = os.path.join(output_path, "res_" + file_name + "_masked.jpg")

        cv2.imwrite(masked_img_path, score_text)

        file_utils.saveResult(file_name, image[:, :, ::-1], polys, dirname=output_path)
"""

def save_outputs(image, region_scores, affinity_scores, text_threshold, link_threshold,
                                           low_text, outoput_path, confidence_mask = None):
    """save image, region_scores, and affinity_scores in a single image. region_scores and affinity_scores must be
    cpu numpy arrays. You can convert GPU Tensors to CPU numpy arrays like this:

    >>> array = tensor.cpu().data.numpy()

    When saving outputs of the network during training, make sure you convert ALL tensors (image, region_score,
    affinity_score) to numpy array first.

    :param image: numpy array
    :param region_scores: [] 2D numpy array with each element between 0~1.
    :param affinity_scores: same as region_scores
    :param text_threshold: 0 ~ 1. Closer to 0, characters with lower confidence will also be considered a word and be boxed
    :param link_threshold: 0 ~ 1. Closer to 0, links with lower confidence will also be considered a word and be boxed
    :param low_text: 0 ~ 1. Closer to 0, boxes will be more loosely drawn.
    :param outoput_path:
    :param confidence_mask:
    :return:
    """

    assert region_scores.shape == affinity_scores.shape
    assert len(image.shape) - 1 == len(region_scores.shape)

    boxes, polys = getDetBoxes(region_scores, affinity_scores, text_threshold, link_threshold,
                                           low_text, False)
    boxes = np.array(boxes, np.int32) * 2
    if len(boxes) > 0:
        np.clip(boxes[:, :, 0], 0, image.shape[1])
        np.clip(boxes[:, :, 1], 0, image.shape[0])
        for box in boxes:
            cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))

    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores)

    if confidence_mask is not None:
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray], axis=0)
        output = np.hstack([image, output])

    else:
        gt_scores = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=0)
        output = np.hstack([image, gt_scores])

    cv2.imwrite(outoput_path, output)
    return output




def save_outputs_from_tensors(images, region_scores, affinity_scores, text_threshold, link_threshold,
                                           low_text, output_dir, image_names, confidence_mask = None):

    """takes images, region_scores, and affinity_scores as tensors (cab be GPU).

    :param images: 4D tensor
    :param region_scores: 3D tensor with values between 0 ~ 1
    :param affinity_scores: 3D tensor with values between 0 ~ 1
    :param text_threshold:
    :param link_threshold:
    :param low_text:
    :param output_dir: direcotry to save the output images. Will be joined with base names of image_names
    :param image_names: names of each image. Doesn't have to be the base name (image file names)
    :param confidence_mask:
    :return:
    """

    #images = images.cpu().permute(0, 2, 3, 1).contiguous().data.numpy()
    if type(images) == torch.Tensor:
        images = np.array(images)

    region_scores = region_scores.cpu().data.numpy()
    affinity_scores = affinity_scores.cpu().data.numpy()

    batch_size = images.shape[0]
    assert batch_size == region_scores.shape[0] and batch_size == affinity_scores.shape[0] and batch_size == len(image_names), \
        "The first dimension (i.e. batch size) of images, region scores, and affinity scores must be equal"

    output_images = []

    for i in range(batch_size):
        image = images[i]
        region_score = region_scores[i]
        affinity_score = affinity_scores[i]

        image_name = os.path.basename(image_names[i])
        outoput_path = os.path.join(output_dir,image_name)

        output_image = save_outputs(image, region_score, affinity_score, text_threshold, link_threshold,
                                           low_text, outoput_path, confidence_mask=confidence_mask)

        output_images.append(output_image)

    return output_images



def make_logger(mode, path=False):

    # mode = iter or epoch

    def logger_path(mode,path):

        if not os.path.exists(f'{path}/{mode}'):
            os.mkdir(f'{path}/{mode}')

        trn_logger_path = os.path.join(f'{path}/{mode}', f'train_{mode}.log')
        val_logger_path = os.path.join(f'{path}/{mode}', f'validation_{mode}.log')
        # val_logger_high_path = os.path.join(f'./saved_models/{opt.experiment_name}/{mode}',
        #                                        f'validation_high_{mode}.log')


        return trn_logger_path, val_logger_path


    trn_logger_path, val_logger_path = logger_path(mode, path)

    trn_logger = writer.Logger(trn_logger_path)
    val_logger = writer.Logger(val_logger_path)
    #val_logger_high = Logger(val_logger_high_path)

    return trn_logger, val_logger


def split_logger(lang_dict):
    eopch_li = []
    loss_li = []
    #acc_li = []
    #ned_li = []

    for i in lang_dict:
        new_dict = i.split()
        eopch_li.append(int(new_dict[0]))
        loss_li.append(float(new_dict[1]))
        #acc_li.append(float(new_dict[2]))
        #ned_li.append(float(new_dict[3]))

    return eopch_li, loss_li

def read_txt(path):
    with open(path, 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]

    eopch_li, loss_li = split_logger(lang_dict)

    return eopch_li, loss_li


# def draw_curve(work_dir, logger1, logger2, val_color, filelname='seg'):
#     logger1 = read_txt(logger1)
#     logger2 = read_txt(logger2)
#
#     epoch, trn_loss1, acc1, ned1 = logger1
#     epoch, trn_loss2, acc2, ned2 = logger2
#
#     plt.figure(1)
#     plt.plot(epoch, trn_loss1, '-b', label='train_loss')
#     plt.plot(epoch, trn_loss2, val_color, label='val_{}_loss'.format(filelname))
#
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.title('compare_loss')
#     plt.savefig(os.path.join(work_dir, 'loss_{}.png'.format(filelname)))
#     plt.close()
#
#     plt.figure(2)
#     plt.plot(epoch, acc1, '-b', label='train_acc')
#     plt.plot(epoch, acc2, val_color, label='val_{}_acc'.format(filelname))
#
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.title('compare_acc')
#     plt.savefig(os.path.join(work_dir, 'val_acc_{}.png'
#                              .format(filelname)))
#
#     plt.close()
#
#     plt.figure(3)
#     plt.plot(epoch, ned1, '-b', label='train_dist')
#     plt.plot(epoch, ned2, val_color, label='val_{}_dist'.format(filelname))
#
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.title('compare_editDist')
#     plt.savefig(os.path.join(work_dir, 'val_1-NED_{}.png'
#                              .format(filelname)))
#
#     plt.close()
#


def draw_curve(work_dir, logger1, logger2, val_color, xlabel='', filelname='seg'):

    logger1 = logger1.read()
    logger2 = logger2.read()

    #logger1 = read_txt(logger1)
    #logger2 = read_txt(logger2)

    epoch, trn_loss1 = zip(*logger1)
    epoch, trn_loss2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, trn_loss1, '-b', label='train_loss')
    plt.plot(epoch, trn_loss2, val_color, label='val_{}_loss'.format(filelname))

    plt.xlabel('{}'.format(xlabel))
    plt.legend()
    plt.title('compare_loss')
    plt.savefig(os.path.join(work_dir, 'loss_{}.png'.format(filelname)))
    plt.close()


def convert_iter_epoch(epoch, iter, len_data):
    if iter == None:
        iter = epoch * len_data
    else:
        if iter % len_data !=0 :
            epoch = iter // len_data + 1
        else:
            epoch = iter // len_data

    return epoch, iter

def save_final_option(args):

    """ final options """
    with open(f'{args.results_dir}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)


def save_model(save_dir,net, optimizer, name, iter, epoch, mode):

    ckpt_path_ep = os.path.join(save_dir,'{}_{}'.format(name,str(mode)) + '.pth')
    torch.save({
        'craft': net.state_dict(),
        'optim': optimizer.state_dict(),
        'step': iter,
        'epoch': epoch
    }, ckpt_path_ep)


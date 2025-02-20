# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import cv2


# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files



# def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None, crop=False):
#         """ save text detection result one by one
#         Args:
#             img_file (str): image file name
#             img (array): raw image context
#             boxes (array): array of result file
#                 Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
#         Return:
#             None
#         """
#
#         img = np.array(img)
#
#         # make result file list
#         filename, file_ext = os.path.splitext(os.path.basename(img_file))
#
#         if not dirname.endswith("/"):
#             dirname = dirname + "/"
#
#         # result directory
#         res_file = dirname + "res_" + filename + '.txt'
#         res_img_file = dirname + "res_" + filename + '.jpg'
#
#         if not os.path.isdir(dirname):
#             os.mkdir(dirname)
#
#         # create directory to save cropped images
#         if crop:
#             cropped_dirname = dirname + "cropped/"
#             if not os.path.exists(cropped_dirname):
#                 os.mkdir(cropped_dirname)
#
#             cropped_dirname = cropped_dirname + filename + "/"
#             if not os.path.exists(cropped_dirname):
#                 os.mkdir(cropped_dirname)
#
#         with open(res_file, 'w') as f:
#             for i, box in enumerate(boxes):
#                 poly = np.array(box).astype(np.int32).reshape((-1))
#                 strResult = ','.join([str(p) for p in poly]) + '\r\n'
#                 f.write(strResult)
#
#                 poly = poly.reshape(-1, 2)
#                 cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
#                 ptColor = (0, 255, 255)
#                 if verticals is not None:
#                     if verticals[i]:
#                         ptColor = (255, 0, 0)
#
#                 if texts is not None:
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     font_scale = 0.5
#                     cv2.putText(img, "{}".format(texts[i]), (poly[0][0] + 1, poly[0][1] + 1), font, font_scale,
#                                 (0, 0, 0), thickness=1)
#                     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255),
#                                 thickness=1)
#
#                 if crop:
#                     cropped_img = Image.open(img_file)
#                     cropped_img_path = cropped_dirname + "cropped" + str(i) + "_" + filename + ".jpg"
#                     left = min(poly[0][0], poly[3][0])
#                     up = min(poly[0][1], poly[1][1])
#                     right = max(poly[1][0], poly[2][0])
#                     down = max(poly[2][1], poly[3][1])
#                     cropped_img = cropped_img.crop((left, up, right, down))
#                     cropped_img.save(cropped_img_path)
#
#         # Save result image
#         cv2.imwrite(res_img_file, img)


def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None, crop=False, gt_bbox=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """

        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        if not dirname.endswith("/"):
            dirname = dirname + "/"

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        gt_file = dirname + "gt_" + filename + '.txt'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # create directory to save cropped images
        if crop:
            cropped_dirname = dirname + "cropped/"
            if not os.path.exists(cropped_dirname):
                os.mkdir(cropped_dirname)

            cropped_dirname = cropped_dirname + filename + "/"
            if not os.path.exists(cropped_dirname):
                os.mkdir(cropped_dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))

                if len(poly) == 8:
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    f.write(strResult)

                    poly = poly.reshape(-1, 2)

                    #gt_poly = np.array(gt_bbox[i]).astype(np.int32)

                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                    #cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 255, 255), thickness=2)

                    ptColor = (0, 255, 255)
                    if verticals is not None:
                        if verticals[i]:
                            ptColor = (255, 0, 0)

                    if texts is not None:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        cv2.putText(img, "{}".format(texts[i]), (poly[0][0] + 1, poly[0][1] + 1), font, font_scale,
                                    (0, 0, 0), thickness=1)
                        cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255),
                                    thickness=1)

                    try:
                        if crop:
                            cropped_img = Image.open(img_file)
                            cropped_img_path = cropped_dirname + "cropped" + str(i) + "_" + filename + ".jpg"
                            left = min(poly[0][0], poly[3][0])
                            up = min(poly[0][1], poly[1][1])
                            right = max(poly[1][0], poly[2][0])
                            down = max(poly[2][1], poly[3][1])
                            cropped_img = cropped_img.crop((left, up, right, down))
                            cropped_img.save(cropped_img_path)
                    except:
                        pass
            f.close()

        if gt_bbox is not None:

            #원하는 파일 뽑기

            import zipfile
            import codecs

            gt_name = "gt_" + filename + '.txt'

            read_file = []
            with zipfile.ZipFile(gt_bbox) as thezip:
                with thezip.open(gt_name, mode='r') as thefile:
                    # Let us verify the operation..
                    read_file.append(thefile.read())


            thefile.close()
            thezip.close()


            raw = codecs.decode(read_file[0], 'utf-8-sig', 'replace')
            raw_li = raw.split('\r\n')

            for gt in raw_li:
                if gt != '':
                    gt_box = gt.split(',')
                    try:
                        gt_box = np.array(list(map(int, gt_box[:8])))
                        gt_poly = gt_box.reshape(-1, 2)
                    except:
                        import ipdb;ipdb.set_trace()

                    gt_poly = np.array(gt_poly).astype(np.int32)
                    cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


            # raw = codecs.decode(read_file[0], 'utf-8-sig', 'replace')
            # raw_li = raw.split('\r\n')
            # with open(gt_file, 'w') as g:
            #     for gt in raw_li:
            #         strResult = gt + '\r\n'
            #         g.write(strResult)
            #         if gt != '':
            #             gt_box = gt.split(',')
            #             try:
            #                 gt_box = np.array(list(map(int, gt_box[:8])))
            #                 gt_poly = gt_box.reshape(-1, 2)
            #             except:
            #                 import ipdb;ipdb.set_trace()
            #
            #             gt_poly = np.array(gt_poly).astype(np.int32)
            #             cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            #
            #g.close()


            #
            # with open(gt_file, 'w') as g:
            #     for i, gt_box in enumerate(zipfile_.open()):
            #         poly = np.array(gt_box).astype(np.int32).reshape((-1))
            #         strResult = ','.join([str(p) for p in poly]) + '\r\n'
            #         g.write(strResult)
            #
            #         gt_poly = np.array(gt_bbox[i]).astype(np.int32)
            #         cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)







        # Save result image
        cv2.imwrite(res_img_file, img)


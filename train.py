import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
import sys
import test
import time

from data_loader.Synth80k import Synth80k
from data_loader.Synth80k_kr import Synth80k_kr

from data_loader.KAIST import KAIST
from data_loader.ICDAR import ICDAR2015, ICDAR2013
import logging
from util.mseloss import Maploss
from collections import OrderedDict
from model.craft import CRAFT
from data_loader.KAIST import KAIST
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset
from util.validate import validate
import numpy as np
import math
from util import craft_utils
from util import writer as craft_writer
from test import validation


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

def adjust_learning_rate(optimizer, lr, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (0.8 ** step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def hierarchical_dataset(root, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []

    print(f'dataset_root:    {root}\t dataset: {select_data[0]}')
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        for i in filenames:
            lmdb_path = os.path.join(dirpath, str(i))
            dataset = Synth80k_kr(lmdb_path, target_size=768)
            print(f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}')
            dataset_list.append(dataset)

    return dataset_list


def train(args, writer, results_dir, images_path):


    ########## load training datawset ##########
    # check command-line inputs
    datasets = []

    if not 'kaist' in args.datasets and not 'synthtext_en' in args.datasets and not 'synthtext_kr' in args.datasets:
        raise ValueError("One of --kaist or --synthtext must be given to specify the dataset used to train.")

    if 'kaist' in args.datasets :

        dataset_kaist = KAIST(args.dataset_path, target_size=768)
        datasets.append(dataset_kaist)
        print(f'sub-directory:\t num samples: {len(dataset_kaist)}')

    if 'synthtext_en' in args.datasets :
        dataset_path_en= '/home/data/ocr/detection/SynthText/SynthText'
        dataset_en = Synth80k(dataset_path_en,target_size=768)
        datasets.append(dataset_en)
        print(f'sub-directory:\t num samples: {len(dataset_en)}')


    if 'synthtext_kr' in args.datasets:
        dataset_path_kr= '/data/workspace/woans0104/SynthText_kr-master/data/background/gen/syn_kr_v2'
        datasets.extend(hierarchical_dataset(dataset_path_kr))



    dataset = ConcatDataset(datasets)



    # split dataset into two subsets: train and valid
    # source: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    indices = list(range(len(dataset)))
    val_split = 0.1
    val_dataset_size = int(np.floor(val_split * len(dataset)))
    val_indices, train_indices = indices[:val_dataset_size], indices[val_dataset_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=val_sampler)

    if args.is_weakly_supervised:
        assert args.aux_dataset_path != "", "Auxilary dataset path is required for weakly supervised training"

        aux_train_loader = torch.utils.data.DataLoader(
            Synth80k(args.dataset_path, target_size=768),
            batch_size=args.batch_size,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            sampler=val_sampler)

        aux_train_loader_iter = iter(aux_train_loader)


    ########## initialize network ##########
    if args.ckpt_path != None:
        ckpt = torch.load(args.ckpt_path)
        if "craft" in list(ckpt.keys()):
            craft_state_dict = ckpt["craft"]
            init_epoch = ckpt["epoch"]
            step = ckpt["step"]
            optim_state_dict = ckpt["optim"]

    else:
        # craft_state_dict = ckpt
        init_epoch = 0
        step = 0
        optim_state_dict = None

    net = CRAFT(use_vgg16_pretrained=args.use_vgg16_pretrained, freeze=args.freeze_vgg16)
    if args.ckpt_path != None:
        net.load_state_dict(craft_state_dict)
    net = torch.nn.DataParallel(net)
    net = net.cuda()


    # initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim_state_dict != None:
        optimizer.load_state_dict(optim_state_dict)
    # initialize loss
    criterion = Maploss()
    # since input images are resized to a fixed size during training, all input sizes are identical
    cudnn.benchmark = True

    # Logger
    trn_logger_ep, val_logger_ep = craft_utils.make_logger(mode='epoch', path=results_dir)
    trn_logger_it, val_logger_it = craft_utils.make_logger(mode='iter', path=results_dir)

    results_dir_ep = f'{results_dir}/epoch'
    results_dir_it = f'{results_dir}/iter'

    best_loss_ep = 1e8
    best_loss_it = 1e8


    """ final options """
    with open(f'{results_dir}/opt.txt', 'a',encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)



    for epoch in range(init_epoch+1, args.epoch+1):

        # restore this after debugging validate
        #if epoch % args.val_epoch_interval == 0 and epoch != 0:
        # if epoch % args.val_epoch_interval == 0:
        #     validate(args, net, val_loader, writer, step, images_path)

        start_time = time.time()
        for images, gh_label, gah_label, mask, _, unnormalized_images, img_paths in tqdm(train_loader):

            net.train()
            step += 1
            losses = craft_writer.AverageMeter()
            #accuracies = craft_writer.AverageMeter()


            # source: https://github.com/clovaai/CRAFT-pytorch/issues/18#issuecomment-513258344
            # initial lr is 1
            # e-4 multiplied by 0.8 for every 10k iterations
            if step % 20000 == 0 and step != 0:
                adjust_learning_rate(optimizer, args.lr, step)

            if args.is_weakly_supervised:
                aux_images, aux_gh_label, aux_gah_label, aux_mask, _, \
                aux_unnormalized_images, aux_img_paths = aux_train_loader_iter

                images = torch.cat((images, aux_images), 0)
                gh_label = torch.cat((gh_label, aux_gh_label), 0)
                gah_label = torch.cat((gah_label, aux_gah_label), 0)
                mask = torch.cat((mask, aux_mask), 0)
                unnormalized_images = torch.cat((unnormalized_images, aux_unnormalized_images), 0)

                for paths in aux_img_paths:
                    img_paths.append(paths)


            images = images.cuda()
            gh_label = gh_label.cuda()
            gah_label = gah_label.cuda()
            mask = mask.cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()

            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()

            # log update
            losses.update(loss.item(), 1)
            #accuracies.update(accuracy, 1)

            # if log explodes, save images and log approriate data
            if loss > 1e8 or math.isnan(loss):
                imgs_paths_str = ""
                for img_path in img_paths:
                    imgs_paths_str += img_path + "\n"

                # log error message
                logging.error("loss %.01f at training step %d!" % (loss, step))
                logging.error("above error occured while processing images:\n" + imgs_paths_str)

                # create path and directories to save imags
                error_images_path = os.path.join(images_path, "train_error")
                if not os.path.exists(error_images_path):
                    os.mkdir(error_images_path)

                output_images_path = os.path.join(error_images_path, "network_" + str(step))
                if not os.path.exists(output_images_path):
                    os.mkdir(output_images_path)

                ref_images_path = os.path.join(error_images_path, "ref_" + str(step))
                if not os.path.exists(ref_images_path):
                    os.mkdir(ref_images_path)

                # save images to disk
                output_images = craft_utils.save_outputs_from_tensors(unnormalized_images, out1, out2,
                                                                      args.text_threshold, args.link_threshold,
                                                                      args.low_text,output_images_path, img_paths)

                ref_images = craft_utils.save_outputs_from_tensors(unnormalized_images, gh_label, gah_label,
                                                                   args.text_threshold, args.link_threshold,
                                                                   args.low_text,ref_images_path, img_paths)


                # log loss and images
                writer.log_output_images(output_images, ref_images, step)
                writer.log_training(loss, step)
                raise Exception("Loss exploded")



            if step % args.val_interval == 0 and step != 0:
                trn_logger_it.write([step, losses.avg])
                val_loss_it = validation(net, val_loader, val_logger_it, step)
                writer.log_training(loss, step)
                # saving the model with loss

                if val_loss_it < best_loss_it:
                    best_loss_it = val_loss_it
                    ckpt_path_it = os.path.join(results_dir_it, str(step) + '.pth')
                    logging.info('Saving ' + ckpt_path_it + ', step:' + str(step))
                    torch.save({
                        'craft': net.state_dict(),
                        'optim': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, ckpt_path_it)





        # log epoch
        elapsed_time = time.time() - start_time
        trn_logger_ep.write([epoch, losses.avg])
        val_loss_ep = validation(net, val_loader, val_logger_ep, epoch)
        print(f'[{args.epoch}/{epoch}] Loss: {losses.avg:0.5f} '
              f'elapsed_time: {elapsed_time:0.5f}')

        if val_loss_ep < best_loss_ep:
            best_loss_ep = val_loss_ep

            ckpt_path_ep = os.path.join(results_dir_ep, str(epoch) + '.pth')
            logging.info('Saving ' + ckpt_path_ep + ', step:' + str(epoch))
            torch.save({
                'craft': net.state_dict(),
                'optim': optimizer.state_dict(),
                'step': epoch,
                'epoch': epoch
            }, ckpt_path_ep)

        # -----------------------------------------------#

    # last model save
    ckpt_path_ep = os.path.join(results_dir_ep, 'lastmodel_{}'.format(str(epoch)) + '.pth')
    logging.info('Saving ' + ckpt_path_ep + ', step:' + str(epoch))
    torch.save({
        'craft': net.state_dict(),
        'optim': optimizer.state_dict(),
        'step': epoch,
        'epoch': epoch
    }, ckpt_path_ep)

    # end

    #test.test(args, ckpt_path)


    craft_utils.draw_curve(results_dir_ep, trn_logger_ep, val_logger_ep, val_color='-r'
                           ,xlabel='epoch', filelname='synthesis')
    craft_utils.draw_curve(results_dir_it, trn_logger_it, val_logger_it, val_color='-r'
                           ,xlabel='iter', filelname='synthesis')

    sys.exit()


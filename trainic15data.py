import os
import torch
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
from test import test
from tqdm import tqdm

from data_loader.Synth80k import Synth80k
from data_loader.ICDAR import ICDAR2015

from util.mseloss import Maploss
from collections import OrderedDict
from eval.script import getresult
from model.craft import CRAFT
from torch.autograd import Variable
from util import craft_utils
from util import writer as craft_writer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from test import validation
# 3.2768e-5
random.seed(42)

parser = argparse.ArgumentParser(description='CRAFT reimplementation')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

parser.add_argument('--results_dir', default=None, type=str,
                    help='Path to save checkpoints')


parser.add_argument('--synth80k_path', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument('--icdar2015_path', default='/home/data/ocr/detection/ICDAR2015', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument("--ckpt_path", default='/data/workspace/woans0104/CRAFT-reimplementation/exp'
                                           '/ICDAR2015_test/lastmodel_1.pth', type=str,
                    help="path to pretrained model")


parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--epoch', default=None, type=int,
                    help='Path to save checkpoints')
parser.add_argument('--iter', default=None, type=int,
                    help='Path to save checkpoints')
parser.add_argument('--val_interval', default=None, type=int,
                    help='Path to save checkpoints')

args = parser.parse_args()


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
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_iter_epoch(epoch, iter, len_data):
    if iter == None:
        iter = epoch * len_data
    else:
        if iter % len_data !=0 :
            epoch = iter // len_data + 1
        else:
            epoch = iter // len_data

    return epoch, iter


if __name__ == '__main__':
    """
    Train with real data from ICDAR and synthetnic data from SynthText within the same batch.

    most of the code is identical to trainSyndata.py.
    """



    # load dataset

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    dataloader = Synth80k(args.synth80k_path, target_size=768)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    batch_syn = iter(train_loader)

    net = CRAFT(use_vgg16_pretrained=False)
    net_param = torch.load(args.ckpt_path)
    try:
        net.load_state_dict(copyStateDict(net_param['craft']))
    except:
        net.load_state_dict(copyStateDict(net_param))

    net = net.cuda()
    net = torch.nn.DataParallel(net)

    cudnn.benchmark = True

    realdata = ICDAR2015(net, args.icdar2015_path, target_size=768)


    indices = list(range(len(realdata)))
    np.random.shuffle(indices)

    val_split = 0.1
    val_dataset_size = int(np.floor(val_split * len(realdata)))
    val_indices, train_indices = indices[:val_dataset_size], indices[val_dataset_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)


    real_data_trn_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=10,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    real_data_val_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=val_sampler)


    # optim & loss

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()


    # Logger
    trn_logger_ep, val_logger_ep = craft_utils.make_logger(mode='epoch', path=args.results_dir)
    trn_logger_it, val_logger_it = craft_utils.make_logger(mode='iter', path=args.results_dir)

    results_dir_ep = f'{args.results_dir}/epoch'
    results_dir_it = f'{args.results_dir}/iter'



    """ final options """
    with open(f'{args.results_dir}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)


    # start train

    epoch, iter = convert_iter_epoch(args.epoch, args.iter,len(real_data_trn_loader))

    step_index = 0
    best_loss_ep = 1e8
    best_loss_it = 1e8
    breaker = True
    for ep in range(epoch):
        train_time_st = time.time()


        start_time = time.time()
        for index, (real_images, real_gh_label, real_gah_label, real_mask, _, _, _) in enumerate(
                tqdm(real_data_trn_loader)):

            net.train()
            step_index += 1
            losses = craft_writer.AverageMeter()

            if step_index % 10000 == 0 and step_index != 0:
                adjust_learning_rate(optimizer, args.lr, step_index)



            # load synthetic images from SythText dataset
            syn_images, syn_gh_label, syn_gah_label, syn_mask, _, _, _ = next(batch_syn)

            # add ICDAR images with synthetic images in the same batch
            images = torch.cat((syn_images, real_images), 0)
            gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            mask = torch.cat((syn_mask, real_mask), 0)

            images = Variable(images.type(torch.FloatTensor)).cuda()
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            # log update
            losses.update(loss.item(), 1)


            #val interval
            if step_index % args.val_interval == 0 and step_index != 0:
                trn_logger_it.write([step_index, losses.avg])
                val_loss_it = validation(net, real_data_val_loader, val_logger_it, step_index)

                # saving the model with loss

                if val_loss_it < best_loss_it:
                    best_loss_it = val_loss_it
                    ckpt_path_it = os.path.join(results_dir_it, str(step_index) + '.pth')
                    torch.save({
                        'craft': net.state_dict(),
                        'optim': optimizer.state_dict(),
                        'step': step_index,
                        'epoch': ep
                    }, ckpt_path_it)


            # last model save_iter
            if step_index == args.iter:
                ckpt_path_it = os.path.join(results_dir_it, 'lastmodel_it_{}'.format(str(step_index)) + '.pth')
                torch.save({
                    'craft': net.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': step_index,
                    'epoch': ep
                }, ckpt_path_it)
                breaker = False
                break

        if breaker == False:
            break


        # log epoch


        elapsed_time = time.time() - start_time
        trn_logger_ep.write([ep, losses.avg])
        print(f'[{epoch}/{ep}] Loss: {losses.avg:0.5f} '
              f'elapsed_time: {elapsed_time:0.5f}')
        val_loss_ep = validation(net, real_data_val_loader, val_logger_ep, ep)

        if val_loss_ep < best_loss_ep:
            best_loss_ep = val_loss_ep

            ckpt_path_ep = os.path.join(results_dir_ep, str(ep) + '.pth')
            torch.save({
                'craft': net.state_dict(),
                'optim': optimizer.state_dict(),
                'step': ep,
                'epoch': ep
            }, ckpt_path_ep)



    # last model save
    ckpt_path_ep = os.path.join(results_dir_ep, 'lastmodel_ep_{}'.format(str(ep)) + '.pth')
    torch.save({
        'craft': net.state_dict(),
        'optim': optimizer.state_dict(),
        'step': ep,
        'epoch': ep
    }, ckpt_path_ep)



    try:
        craft_utils.draw_curve(results_dir_ep, trn_logger_ep, val_logger_ep, val_color='-r'
                               , xlabel='epoch', filelname='synthesis')
        craft_utils.draw_curve(results_dir_it, trn_logger_it, val_logger_it, val_color='-r'
                               , xlabel='iter', filelname='synthesis')

    except Exception as e:
        print(e)

    print('end training')









# source: https://github.com/seungwonpark/melgan/blob/master/utils/writer.py
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

from tensorboardX import SummaryWriter
from collections import Iterable



class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log





class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)

        else:
            pass


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)
        self.is_first = True

    def log_training(self, loss, step):
        self.add_scalar('train.loss', loss, step)

    def log_validation(self, loss, net, step, my_outputs=None, ref_outputs=None, save_images=False):
        logging.info("step = " + str(step) + ", loss = " + str(loss))

        self.add_scalar('validation.loss', loss, step)
        self.log_histogram(net, step)

        if save_images:
            self.log_output_images(my_outputs, ref_outputs, step)

    def log_output_images(self, my_outputs, ref_outputs, step):
        """Inputs must be the returned list of images from `craft_util.save_outputs_from_tensors`.

        :param my_outputs:
        :param ref_outputs:
        :param step:
        :return:
        """

        batch_size = len(my_outputs)
        assert batch_size == len(ref_outputs)

        for i in range(batch_size):
            self.add_image("output_image", my_outputs[i], step, dataformats="HWC")
            self.add_image("ref_image", ref_outputs[i], step, dataformats="HWC")

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
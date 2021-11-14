import os
import time
import shutil
import torch
import json
import numpy as np
import vsrn_data3 as data
from vsrn_model import FSCMM
from collections import OrderedDict
import logging
import tensorboard_logger as tb_logger
import argparse
from vsrn_evaluation import evalrank


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)

        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    We use the open codes offered by
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    We use the open codes offered by
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def use_memory(model, data_loader, log_step=10):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval_mode()
    end = time.time()

    v_features = None
    t_features = None
    cap_lens = None
    max_n_word = 0

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        model.logger = val_logger
        with torch.no_grad():
            img_feature, cap_feature, v_enhanced, t_enhanced, cap_len, v_weight, t_weight = model.forward_mem(images, captions, lengths, volatile=True)

        if v_features is None:
            if v_enhanced.dim() == 3:
                v_features = np.zeros((len(data_loader.dataset), v_enhanced.size(1), v_enhanced.size(2)))
            else:
                v_features = np.zeros((len(data_loader.dataset), v_enhanced.size(1)))
            t_features = np.zeros((len(data_loader.dataset), max_n_word, t_enhanced.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        v_features[ids] = v_enhanced.data.cpu().numpy().copy()
        t_features[ids, :max(lengths), :] = t_enhanced.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        model.forward_loss(v_enhanced, t_enhanced, cap_len)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            print('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)))
        del images, captions
    return v_features, t_features, cap_lens


def train(opt, train_loader, model, epoch, val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    end = time.time()

    for i, train_data in enumerate(train_loader):
        model.train_mode()
        data_time.update(time.time() - end)

        model.logger = train_logger

        model.train_step(*train_data)

        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Path: {path}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time, data_time=data_time, e_log=str(model.logger), path=opt.info_save_path + 'log')
            )

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        if model.Eiters % opt.val_step == 0:
            evalrank(model, val_loader, opt, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def save_memory(v_memory, t_memory, is_best, filename='memory.npy', prefix=''):
    tries = 15
    error = None

    while tries:
        try:
            v_memory = v_memory.detach().cpu().numpy()
            t_memory = t_memory.detach().cpu().numpy()
            np.set_printoptions(threshold=np.inf)
            np.save(prefix + 'v_' + filename, v_memory)
            np.save(prefix + 't_' + filename, t_memory)
            if is_best:
                shutil.copyfile(prefix + 'v_' + filename, prefix + 'v_memory_best.npy')
                shutil.copyfile(prefix + 't_' + filename, prefix + 't_memory_best.npy')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('memory save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.init_lr * (opt.lr_rate ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--notes', default='', type=str, help='notes of this run')
    parser.add_argument('--dataset', default='AI2D#', type=str, help='flickr30k/MSCOCO/AI2D#')
    parser.add_argument('--data_mode', default='train', type=str, help='train/test')
    parser.add_argument('--no_fragment_attn', action='store_true', help='whether to use local fragment attention mechanism')
    parser.add_argument('--lambda_txt_weight', default=1., type=float, help='hyper params of text-to-vision alignment')
    parser.add_argument('--info_save_path', default='', help='path to save info and results')
    parser.add_argument('--no_precomp', action='store_true', help='type of data')
    parser.add_argument('--no_self_regulating', action='store_true', help='whether to use self_regulating memory or normal memory')
    parser.add_argument('--regulate_way', default='clamp', type=str, help='self-regulating type: sigmoid/clamp/softmax')
    parser.add_argument('--new_opt', action='store_true', help='whether to use new opt information when resume training')
    parser.add_argument('--margin', default=0.2, type=float, help='ranking loss margin')
    parser.add_argument('--init_lr', default=.0002, type=float, help='initial learning rate')
    parser.add_argument('--lr_update', default=10, type=int, help='number of epochs to update the learning rate')
    parser.add_argument('--lr_rate', default=0.1, type=float, help='decay rate of learning rate')
    parser.add_argument('--num_epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='size of a training mini-batch')
    parser.add_argument('--memory_dim', default=1024, type=int, help='dimensionality of the memory')
    parser.add_argument('--word_dim', default=768, type=int, help='dimensionality of the initial word embedding')
    parser.add_argument('--embed_size', default=1024, type=int, help='dimensionality of the joint embedding space')
    parser.add_argument('--cross_attn', default="both", help='t2i|i2t|both')
    parser.add_argument('--memory_size', default=200, type=int, help='number of slots of the memory')
    parser.add_argument('--loss_detail', action='store_true', help='whether to log eiter loss')
    parser.add_argument('--log_step', default=50, type=int, help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=999999, type=int, help='number of steps to run validation')
    parser.add_argument('--grad_clip', default=2., type=float, help='gradient clipping threshold')
    parser.add_argument('--num_layers', default=1, type=int, help='number of GRU/bi-GRU layers')
    parser.add_argument('--workers', default=10, type=int, help='number of data loader workers')
    parser.add_argument('--resume_file', default='', type=str, metavar='PATH', help='resume from, for example: checkpoint_1')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--img_dim', default=2048, type=int, help='dimensionality of the initial image embedding')
    parser.add_argument('--no_img_norm', action='store_true', help='do not normalize the image embeddings')
    parser.add_argument('--no_txt_norm', action='store_true', help='do not normalize the text embeddings')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm", help='clipped_l2norm|l2norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp", help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--use_gru', action='store_true', help='use GRU to encode word representations')
    parser.add_argument('--bi_gru', action='store_true', help='enable bi-GRU')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temp')
    parser.add_argument('--lambda_softmax', default=9., type=float, help='attention softmax temperature')
    parser.add_argument('--use_abs', action='store_true', help='take the absolute value of embedding vectors')
    parser.add_argument("--max_len", type=int, default=60, help='max length of captions(containing <sos>,<eos>)')
    parser.add_argument('--iteration_step', default=1, type=int, help='')

    opt = parser.parse_args()
    print(opt)

    if opt.resume_file:
        if os.path.isfile('./checkpoint/' + opt.resume_file + '.pth.tar'):
            print("=> getting info path and dataset from '{}'".format('./checkpoint/' + opt.resume_file))
            checkpoint = torch.load('./checkpoint/' + opt.resume_file + '.pth.tar')
            opt_temp = checkpoint['opt']
            opt_new = opt
            opt = opt_temp
            opt.resume_file = opt_new.resume_file
            opt.new_opt = opt_new.new_opt

    data_name = opt.dataset
    crop_size = 224
    vocab = 'no_vocab'

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.info_save_path + 'log', flush_secs=5)
    with open(opt.info_save_path + 'opt_info.txt', 'a') as f:
        f.writelines(str(opt) + '\n\n')

    train_loader, val_loader = data.get_loaders(data_name, vocab, opt.batch_size, opt.workers, opt)

    model = FSCMM(opt)

    best_rsum = 0
    start_epoch = 0
    if opt.resume_file:
        if os.path.isfile(opt.info_save_path + 'checkpoint/' + opt.resume_file + '.pth.tar'):
            print("=> loading checkpoint from '{}'".format(opt.info_save_path + opt.resume_file))
            checkpoint = torch.load(opt.info_save_path + 'checkpoint/' + opt.resume_file + '.pth.tar')
            v_memory = np.load(opt.info_save_path + 'checkpoint/' + opt.resume_file.replace('checkpoint', 'v_memory') + '.npy')
            t_memory = np.load(opt.info_save_path + 'checkpoint/' + opt.resume_file.replace('checkpoint', 't_memory') + '.npy')
            opts = checkpoint['opt']
            opts.vocab_size = opt.vocab_size
            if opt.new_opt:
                opts.num_epochs = opt_new.num_epochs
                opts.lr_update = opt_new.lr_update
                opts.init_lr = opt_new.init_lr
            print(opts)
            model = FSCMM(opts, v_memory, t_memory)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opts.info_save_path + 'checkpoint/' + opt.resume_file, start_epoch, best_rsum))
            evalrank(model, val_loader, opts, opt.resume_file.split('_')[-1])
        else:
            print("=> no checkpoint found at '{}'".format(opt.info_save_path + 'checkpoint/'))
        opt = opts

    for epoch in range(start_epoch, opt.num_epochs):
        print("Saving info into >>>> ", opt.info_save_path)
        print('Model setting >>>> ', opt.notes)

        adjust_learning_rate(opt, model.optimizer, epoch)
        adjust_learning_rate(opt, model.optimizer_memory, epoch)

        train(opt, train_loader, model, epoch, val_loader)

        rsum = evalrank(model, val_loader, opt, str(epoch))

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.info_save_path + 'checkpoint/'):
            os.mkdir(opt.info_save_path + 'checkpoint/')
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.info_save_path + 'checkpoint/')
        save_memory(model.cm_memory.v_memory, model.cm_memory.t_memory, is_best,
                    filename='memory_{}.npy'.format(epoch), prefix=opt.info_save_path + 'checkpoint/')

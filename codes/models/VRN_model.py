import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class VRNModel(BaseModel):
    def __init__(self, opt):
        super(VRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w]).cuda())
            c_t.append(torch.zeros([b, c, h, w]).cuda())
        memory = torch.zeros([b, c, h, w]).cuda()

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        if self.opt['model'] == 'LSTM-VRN':
            l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
            return l_forw_fit
        elif self.opt['model'] == 'MIMO-VRN':
            l_forw_fit = 0
            for i in range(out.shape[1]):
                l_forw_fit += self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, i], y[:, i])
            return l_forw_fit

    def loss_back_rec(self, out, x):
        if self.opt['model'] == 'LSTM-VRN':
            l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
            return l_back_rec
        elif self.opt['model'] == 'MIMO-VRN':
            l_back_rec = 0
            for i in range(x.shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[:, i], x[:, i])
            return l_back_rec

    def loss_center(self, out, x):
        # x.shape: (b, t, c, h, w)
        b, t = x.shape[:2]
        l_center = 0
        for i in range(b):
            mse_s = self.Reconstruction_center(out[i], x[i])
            mse_mean = torch.mean(mse_s)
            for j in range(t):
                l_center += torch.sqrt((mse_s[j] - mse_mean.detach()) ** 2 + 1e-18)
        l_center = self.train_opt['lambda_center'] * l_center / b

        return l_center

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()

        if self.opt['model'] == 'LSTM-VRN':
            # forward downscaling
            b, t, c, h, w = self.real_H.shape
            self.output = [self.netG(x=self.real_H[:, i]) for i in range(t)]

            # hidden state initialization
            z_p = torch.zeros(self.output[0][:, 3:].shape).to(self.device)
            hs = self.init_hidden_state(z_p)
            z_p_back = torch.zeros(self.output[0][:, 3:].shape).to(self.device)
            hs_back = self.init_hidden_state(z_p_back)

            # LSTM forward
            for i in range(self.center + 1):
                y = self.Quantization(self.output[i][:, :3])
                z_p, hs = self.netG(x=[y, z_p], rev=True, hs=hs, direction='f')
            # LSTM backward
            for j in reversed(range(self.center, t)):
                y = self.Quantization(self.output[j][:, :3])
                z_p_back, hs_back = self.netG(x=[y, z_p_back], rev=True, hs=hs_back, direction='b')

            # backward upscaling
            y = self.Quantization(self.output[self.center][:, :3])
            out_x, out_z = self.netG(x=[y, [z_p, z_p_back]], rev=True)

            l_back_rec = self.loss_back_rec(self.real_H[:, self.center], out_x)
            LR_ref = self.ref_L[:, self.center].detach()
            l_forw_fit = self.loss_forward(self.output[self.center][:, :3], LR_ref)

            # total loss
            loss = l_forw_fit + l_back_rec
            loss.backward()

        elif self.opt['model'] == 'MIMO-VRN':
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2

            self.input = self.real_H[:, center - intval:center + intval + 1]
            self.output = self.netG(x=self.input.reshape(b, -1, h, w))

            LR_ref = self.ref_L[:, center - intval:center + intval + 1].detach()
            out_lrs = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h // 4, w // 4)
            l_forw_fit = self.loss_forward(out_lrs, LR_ref)

            y = self.Quantization(self.output[:, :3 * self.gop, :, :])
            out_x, out_z = self.netG(x=[y, None], rev=True)

            l_back_rec = self.loss_back_rec(out_x.reshape(-1, self.gop, 3, h, w), self.input)
            l_center_x = self.loss_center(out_x.reshape(-1, self.gop, 3, h, w), self.input)

            # total loss
            loss = l_forw_fit + l_back_rec + l_center_x
            loss.backward()

            if self.train_opt['lambda_center'] != 0:
                self.log_dict['l_center_x'] = l_center_x.item()
        else:
            raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

        # set log
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

    def test(self):
        Lshape = self.ref_L.shape

        self.netG.eval()
        with torch.no_grad():

            if self.opt['model'] == 'LSTM-VRN':

                forw_L = []
                fake_H = []
                b, t, c, h, w = self.real_H.shape

                # forward downscaling
                self.output = [self.netG(x=self.real_H[:, i]) for i in range(t)]

                for i in range(t):
                    # hidden state initialization
                    z_p = torch.zeros(self.output[0][:, 3:].shape).to(self.device)
                    hs = self.init_hidden_state(z_p)
                    z_p_back = torch.zeros(self.output[0][:, 3:].shape).to(self.device)
                    hs_back = self.init_hidden_state(z_p_back)

                    # find sequence index
                    if i - self.center < 0:
                        indices_past = [0 for _ in range(self.center - i)]
                        for index in range(i + 1):
                            indices_past.append(index)
                        indices_future = [index for index in range(i, i + self.center + 1)]
                    elif i > t - self.center - 1:
                        indices_past = [index for index in range(i - self.center, i + 1)]
                        indices_future = [index for index in range(i, t)]
                        for index in range(self.center - len(indices_future) + 1):
                            indices_future.append(t - 1)
                    else:
                        indices_past = [index for index in range(i - self.center, i + 1)]
                        indices_future = [index for index in range(i, i + self.center + 1)]

                    # LSTM forward
                    for j in indices_past:
                        y = self.Quantization(self.output[j][:, :3])
                        z_p, hs = self.netG(x=[y, z_p], rev=True, hs=hs, direction='f')
                    # LSTM backward
                    for k in reversed(indices_future):
                        y = self.Quantization(self.output[k][:, :3])
                        z_p_back, hs_back = self.netG(x=[y, z_p_back], rev=True, hs=hs_back, direction='b')

                    # backward upscaling
                    y = self.Quantization(self.output[i][:, :3])
                    out_x, out_z = self.netG(x=[y, [z_p, z_p_back]], rev=True)

                    forw_L.append(y)
                    fake_H.append(out_x)

            elif self.opt['model'] == 'MIMO-VRN':

                forw_L = []
                fake_H = []
                b, t, c, h, w = self.real_H.shape
                n_gop = t // self.gop

                for i in range(n_gop + 1):
                    if i == n_gop:
                        # calculate indices to pad last frame
                        indices = [i * self.gop + j for j in range(t % self.gop)]
                        for _ in range(self.gop - t % self.gop):
                            indices.append(t - 1)
                        self.input = self.real_H[:, indices]
                    else:
                        self.input = self.real_H[:, i * self.gop:(i + 1) * self.gop]

                    # forward downscaling
                    self.output = self.netG(x=self.input.reshape(b, -1, h, w))
                    out_lrs = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h // 4, w // 4)

                    # backward upscaling
                    y = self.Quantization(self.output[:, :3 * self.gop, :, :])
                    out_x, out_z = self.netG(x=[y, None], rev=True)
                    out_x = out_x.reshape(-1, self.gop, 3, h, w)

                    if i == n_gop:
                        for j in range(t % self.gop):
                            forw_L.append(out_lrs[:, j])
                            fake_H.append(out_x[:, j])
                    else:
                        for j in range(self.gop):
                            forw_L.append(out_lrs[:, j])
                            fake_H.append(out_x[:, j])

            else:
                raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

            self.fake_H = torch.stack(fake_H, dim=1)
            self.forw_L = torch.stack(forw_L, dim=1)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_util import initialize_weights_xavier
from .st_lstm import SpatioTemporalLSTMCell


class ResidualBlockNoBN(nn.Module):
	def __init__(self, nf=64, model='MIMO-VRN'):
		super(ResidualBlockNoBN, self).__init__()
		self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		# honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
		# but this is how we trained the model in the first place and what we reported in the paper
		if model == 'LSTM-VRN':
			self.relu = nn.ReLU(inplace=True)
		elif model == 'MIMO-VRN':
			self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# initialization
		initialize_weights_xavier([self.conv1, self.conv2], 0.1)

	def forward(self, x):
		identity = x
		out = self.relu(self.conv1(x))
		out = self.conv2(out)
		return identity + out


class InvBlockExp(nn.Module):
	def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
		super(InvBlockExp, self).__init__()

		self.split_len1 = channel_split_num
		self.split_len2 = channel_num - channel_split_num

		self.clamp = clamp

		self.F = subnet_constructor(self.split_len2, self.split_len1)
		self.G = subnet_constructor(self.split_len1, self.split_len2)
		self.H = subnet_constructor(self.split_len1, self.split_len2)

	def forward(self, x, rev=False):
		x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

		if not rev:
			y1 = x1 + self.F(x2)
			self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
			y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
		else:
			self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
			y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
			y1 = x1 - self.F(y2)

		return torch.cat((y1, y2), 1)

	def jacobian(self, x, rev=False):
		if not rev:
			jac = torch.sum(self.s)
		else:
			jac = -torch.sum(self.s)

		return jac / x.shape[0]


class HaarDownsampling(nn.Module):
	def __init__(self, channel_in):
		super(HaarDownsampling, self).__init__()
		self.channel_in = channel_in

		self.haar_weights = torch.ones(4, 1, 2, 2)

		# H
		self.haar_weights[1, 0, 0, 1] = -1
		self.haar_weights[1, 0, 1, 1] = -1

		# V
		self.haar_weights[2, 0, 1, 0] = -1
		self.haar_weights[2, 0, 1, 1] = -1

		# D
		self.haar_weights[3, 0, 1, 0] = -1
		self.haar_weights[3, 0, 0, 1] = -1

		self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
		self.haar_weights = nn.Parameter(self.haar_weights)
		self.haar_weights.requires_grad = False

	def forward(self, x, rev=False):
		if not rev:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)

			out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
			out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
			return out
		else:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)

			out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
			return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

	def jacobian(self, x, rev=False):
		return self.last_jac


class InvNN(nn.Module):
	def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
		super(InvNN, self).__init__()

		operations = []

		current_channel = channel_in
		for i in range(down_num):
			b = HaarDownsampling(current_channel)
			operations.append(b)
			current_channel *= 4
			for j in range(block_num[i]):
				b = InvBlockExp(subnet_constructor, current_channel, channel_out)
				operations.append(b)

		self.operations = nn.ModuleList(operations)

	def forward(self, x, rev=False, cal_jacobian=False):
		out = x
		jacobian = 0

		if not rev:
			for op in self.operations:
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		else:
			for op in reversed(self.operations):
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)

		if cal_jacobian:
			return out, jacobian
		else:
			return out


class PredictiveModuleLSTM(nn.Module):
	def __init__(self, channel_in, nf, block_num_rbm=4, model='MIMO-VRN'):
		super(PredictiveModuleLSTM, self).__init__()

		self.block_num_rbm = block_num_rbm
		rbm = []
		y_ext = []
		att = []
		for _ in range(block_num_rbm):
			rbm.append(SpatioTemporalLSTMCell(nf, nf, 3, 1))
			y_ext.append(ResidualBlockNoBN(nf, model))
			att.append(nn.Conv2d(nf, nf, 1, 1, 0, bias=True))
		self.rbm = nn.ModuleList(rbm)
		self.y_ext = nn.ModuleList(y_ext)
		self.att = nn.ModuleList(att)

		self.y_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)

		initialize_weights_xavier(self.y_in)
		initialize_weights_xavier(self.y_ext)
		initialize_weights_xavier(self.att)

	def forward(self, z, h_t, c_t, mem, y_t, rev=False):
		# z: output from previous timestep
		# y_t: current LR
		# x, h_t, c_t: list of output at each hidden states

		y_t = self.y_in(y_t)

		y_t = self.y_ext[0](y_t)
		h_t[0], c_t[0], mem = self.rbm[0](z, h_t[0], c_t[0], mem)
		# attention module
		g = torch.sigmoid(self.att[0](y_t))
		h_t[0] = (1 - g) * h_t[0] + g * y_t

		for i in range(1, self.block_num_rbm):
			h_t[i], c_t[i], mem = self.rbm[i](h_t[i - 1], h_t[i], c_t[i], mem)
			y_t = self.y_ext[i](y_t)
			# attention module
			g = torch.sigmoid(self.att[i](y_t))
			h_t[i] = (1 - g) * h_t[i] + g * y_t

		return h_t, c_t, mem


class PredictiveModuleMIMO(nn.Module):
	def __init__(self, channel_in, nf, block_num_rbm=8):
		super(PredictiveModuleMIMO, self).__init__()

		self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
		residual_block = []
		for i in range(block_num_rbm):
			residual_block.append(ResidualBlockNoBN(nf))
		self.residual_block = nn.Sequential(*residual_block)

	def forward(self, x):
		x = self.conv_in(x)
		return self.residual_block(x)


class Net(nn.Module):
	def __init__(self, opt, subnet_constructor=None, down_num=2):
		super(Net, self).__init__()

		self.model = opt['model']
		opt_net = opt['network_G']

		if self.model == 'LSTM-VRN':
			self.channel_in = opt_net['in_nc']
			self.channel_out = opt_net['out_nc']
		elif self.model == 'MIMO-VRN':
			self.gop = opt['gop']
			self.channel_in = opt_net['in_nc'] * self.gop
			self.channel_out = opt_net['out_nc'] * self.gop
		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

		self.block_num = opt_net['block_num']
		self.block_num_rbm = opt_net['block_num_rbm']
		self.nf = self.channel_in * 4 ** down_num - self.channel_in
		self.irn = InvNN(self.channel_in, self.channel_out, subnet_constructor, self.block_num, down_num)

		if self.model == 'LSTM-VRN':
			self.pm = PredictiveModuleLSTM(self.channel_in, self.nf, self.block_num_rbm, self.model)
			self.pm_back = PredictiveModuleLSTM(self.channel_in, self.nf, self.block_num_rbm, self.model)
			self.z_comb = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True)
			initialize_weights_xavier(self.z_comb)
		elif self.model == 'MIMO-VRN':
			self.pm = PredictiveModuleMIMO(self.channel_in, self.nf)
		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

	def forward(self, x, rev=False, hs=[], direction='f'):

		if self.model == 'LSTM-VRN':
			if not rev:
				# forward upscaling
				out_y = self.irn(x, rev)

				# out_y: downscaled LR output
				return out_y
			else:
				if not hs:
					# backward upscaling
					y, z_pred = x
					z_pred = self.z_comb(torch.cat(z_pred, dim=1))
					out_x = self.irn(torch.cat([y, z_pred], dim=1), rev)

					# out_x: upscaled HR output
					# z_pred: predicted high frequency component z
					return out_x, z_pred
				else:
					# LSTM forward/backward propagation
					y, z_p = x
					h_t, c_t, memory = hs
					if direction == 'f':
						h_t, c_t, memory = self.pm(z_p, h_t, c_t, memory, y, rev)
					else:
						h_t, c_t, memory = self.pm_back(z_p, h_t, c_t, memory, y, rev)
					out_z = h_t[-1]

					# out_z: lstm propagated hidden information
					# [...]: hidden state
					return out_z, [h_t, c_t, memory]

		elif self.model == 'MIMO-VRN':
			if not rev:
				out_y = self.irn(x, rev)

				# out_y: downscaled LR output
				return out_y
			else:
				y, _ = x

				out_z = self.pm(y)
				out_x = self.irn(torch.cat([y, out_z], dim=1), rev)

				# out_x: upscaled HR output
				# out_z: predicted high frequency component z
				return out_x, out_z

		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

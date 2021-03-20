import torch
import torch.nn as nn
import torch.nn.functional as f
from .module_util import initialize_weights_xavier


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding)

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

        initialize_weights_xavier([self.conv_x, self.conv_h, self.conv_m, self.conv_o, self.conv_last])

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        x_concat = f.layer_norm(x_concat, x_concat.size()[1:])
        h_concat = self.conv_h(h_t)
        h_concat = f.layer_norm(h_concat, h_concat.size()[1:])
        m_concat = self.conv_m(m_t)
        m_concat = f.layer_norm(m_concat, m_concat.size()[1:])
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

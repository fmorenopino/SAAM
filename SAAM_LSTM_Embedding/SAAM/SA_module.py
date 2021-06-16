import torch
from torch import nn
import torch.nn.functional as F


class SA_module(nn.Module):


    #def __init__(self, lstm_dim, T_f):
    def __init__(self, lstm_dim, T_f, FFT_dim):
        super(SA_module, self).__init__()

        self.T_f = T_f
        self.FFT_dim = FFT_dim
        self.lstm_dim = lstm_dim
        self.attn_l = nn.Linear(self.lstm_dim * 2, self.lstm_dim)
        self.sig = nn.Sigmoid()
        self.tanh = torch.tanh
        self.attn_g_dim_change = nn.Linear(self.FFT_dim, self.lstm_dim)
        self.attn_g = nn.Linear(self.lstm_dim * 2, self.lstm_dim)

    def forward(self, h, PSD, H):
        H_padd = (F.pad((H.permute(1, -1, 0)), pad=(self.T_f - H.shape[0], 0))).permute(-1, 0, 1)
        fft = torch.rfft(H_padd, signal_ndim=3, onesided=False)
        fft_orig = fft.clone()

        h = h.unsqueeze(1).repeat(1, self.T_f, 1)
        s_l = fft[:,:,:,0].permute(1,0,-1)
        energy_l = self.tanh(self.attn_l(torch.cat((h, s_l), dim=2)))
        alpha_l = self.sig(energy_l)
        tmp_l = alpha_l * fft[:, :, :, 0].data.permute(1, 0, -1)

        #Global
        s_g = PSD.repeat(h.shape[0], h.shape[1], 1)
        s_g = self.attn_g_dim_change(s_g)
        energy_g = self.tanh(self.attn_g(torch.cat((h, s_g), dim=2)))
        alpha_g = self.sig(energy_g)
        tmp_g = alpha_g * s_g
        fft[:, :, :, 0] = tmp_l.permute(1,0,-1) + tmp_g.permute(1,0,-1)

        ifft = torch.ifft(fft, signal_ndim=3)
        H_filtered=ifft[:, :, :, 0]
        H_filtered_ = ifft[:, :, :, 1]

        h_t_filtered = H_filtered[-1, :, :]
        return h_t_filtered, H_filtered.cpu().data.numpy(), alpha_l.permute(0,-1,1), fft_orig, tmp_l.permute(0,-1,1)

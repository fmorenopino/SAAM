"""
@fmorenopino

Frequency Attentive Module
"""

import torch
from torch import nn
import torch.nn.functional as F

class SA_module(nn.Module):


    def __init__(self, emb_dim, T_f, input_size, output_size, FFT_dim):
        super(SA_module, self).__init__()

        self.T_f = T_f
        self.emb_dim = emb_dim
        self.attn_l = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.sig = nn.Sigmoid()
        self.tanh = torch.tanh

        self.FFT_dim = FFT_dim
        self.attn_g_dim_change = nn.Linear(self.FFT_dim, self.emb_dim)
        self.attn_g = nn.Linear(self.emb_dim * 2, self.emb_dim)


    def forward(self, h):
        H = h.permute(1,0,-1).clone()

        Rx = self.get_Rx(H.permute(1, 0, 2))
        PSD = self.get_FFT_(Rx)
        PSD = (F.pad(PSD, pad=(self.FFT_dim - PSD.shape[0], 0)))

        fft = torch.rfft(H, signal_ndim=3, onesided=False)

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
        H_filtered=ifft[:, :, :, 0].permute(1,0,-1)

        return H_filtered

    def get_Rx(self, x):
        """
        On this version the Rx is obtained throught the DFT by: R_x = IFFT(F*F_conj), being F = RFFT(x).
        This is much faster than previous implementations.

        :param x: batch of signals we want to obtain the Rx from
        :return: E{Rx}
        """

        B = x.shape[0]
        F = x.shape[1]
        L = x.shape[-1]
        size = B*F*L

        A = torch.rfft(x, signal_ndim=2, onesided=False)
        S = torch.conj(A) * A / size
        #S = torch.from_numpy(np.conj(A.data_.cpu().numpy())).to(self.params.device) * A / size

        c_fourier = torch.ifft(S, signal_ndim=2)
        E_Rx = torch.mean(c_fourier[:,:,0,1], 0)

        return E_Rx

    def get_FFT_(self, E_Rx):
        """

        :param E_Rx: Expectation of the Autocorrelation(E{Rx}) of dimensions (L).
        :return:
        """

        FFT = torch.rfft(E_Rx, 1)[:, 0]
        FFT[0] = 0

        return FFT



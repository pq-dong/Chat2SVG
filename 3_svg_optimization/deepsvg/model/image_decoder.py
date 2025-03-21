import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageDecoder(nn.Module):
    def __init__(self, img_size, input_nc, output_nc, ngf=16, norm_layer=nn.LayerNorm):

        super(ImageDecoder, self).__init__()
        n_upsampling = int(math.log(img_size, 2))
        ks_list = [3] * (n_upsampling // 3) + [5] * \
            (n_upsampling - n_upsampling // 3)
        stride_list = [2] * n_upsampling
        decoder = []

        chn_mult = []
        for i in range(n_upsampling):
            chn_mult.append(2 ** (n_upsampling - i - 1))

        decoder += [nn.ConvTranspose2d(input_nc, chn_mult[0] * ngf,
                                       kernel_size=ks_list[0], stride=stride_list[0],
                                       padding=ks_list[0] // 2, output_padding=stride_list[0]-1),
                    norm_layer([chn_mult[0] * ngf, 2, 2]),
                    nn.ReLU(True)]

        for i in range(1, n_upsampling):  # add upsampling layers
            chn_prev = chn_mult[i - 1] * ngf
            chn_next = chn_mult[i] * ngf
            decoder += [nn.ConvTranspose2d(chn_prev, chn_next, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2, output_padding=stride_list[i]-1),
                        norm_layer([chn_next, 2 ** (i+1), 2 ** (i+1)]),
                        nn.ReLU(True)]

        decoder += [nn.Conv2d(chn_mult[-1] * ngf, output_nc,
                              kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

    def forward(self, latent_feat, trg_char=None, trg_img=None):
        """Standard forward"""
        # dec_input = torch.cat((latent_feat, trg_char),-1)
        dec_input = latent_feat
        dec_input = dec_input.view(dec_input.size(0), dec_input.size(1), 1, 1)
        dec_out = self.decode(dec_input)
        output = {}
        output['gen_imgs'] = dec_out
        if trg_img is not None:
            output['img_l1loss'] = F.l1_loss(dec_out, trg_img)

        return output

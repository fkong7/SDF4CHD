import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class IsenseeContextModule(nn.Module):
    def __init__(self, in_channels, num_filters, stride=1, dropout_rate=0.3):

        super(IsenseeContextModule, self).__init__()

        self.conv3d_0 = self.conv_norm_lrelu(in_channels, num_filters, stride=stride)
        self.conv3d_1 = self.conv_norm_lrelu(num_filters, num_filters, stride=1)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.conv3d_2 = self.conv_norm_lrelu(num_filters, num_filters, stride=1)

    def conv_norm_lrelu(self, num_feat_in, num_feat_out, stride):
        return nn.Sequential(
            nn.Conv3d(num_feat_in, num_feat_out, kernel_size=3, stride=stride, padding='same' if stride==1 else (1,1,1)),
            nn.InstanceNorm3d(num_feat_out),
            nn.LeakyReLU())

    def forward(self, input):
        conv0 = self.conv3d_0(input)
        conv1 = self.dropout3d(self.conv3d_1(conv0))
        conv2 = (self.conv3d_2(conv1) + conv0)/2.
        return conv2

class Isensee3DUNetEncoder(nn.Module):
    def __init__(self, in_channels, base_n_filter = 16, z_dim=32, n_conv_blocks=4):
        super(Isensee3DUNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.base_n_filter = base_n_filter
        self.z_dim = z_dim
        self.n_conv_blocks = n_conv_blocks

        in_filter = self.in_channels
        for i in range(n_conv_blocks):
            out_filter = self.base_n_filter * 2**(i)
            setattr(self, 'conv_block{}'.format(i), IsenseeContextModule(in_filter, out_filter, stride=1 if i==0 else 2))
            setattr(self, 'conv1_block{}'.format(i), nn.Conv3d(out_filter, z_dim, 3, stride=1, padding=1))
            setattr(self, 'conv2_block{}'.format(i), nn.Conv3d(z_dim, z_dim, 3, stride=1, padding=1))
            in_filter = out_filter

    def forward(self, input):
        feat_list = []
        x = input
        x = F.avg_pool3d(x, kernel_size=4)
        for i in range(self.n_conv_blocks):
            x = getattr(self, 'conv_block{}'.format(i))(x)
            x_out = F.leaky_relu(getattr(self, 'conv1_block{}'.format(i))(x), negative_slope=0.02, inplace=True)
            x_out = getattr(self, 'conv2_block{}'.format(i))(x_out)
        feat_vol = x_out
        return feat_vol



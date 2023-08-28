"""
Channel and Spatial CSNet Network (CS-Net).
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
#######################################################################################
nonlinearity = partial(F.relu, inplace=True)

class MultiTaskNet(nn.Module):
    def __init__(self, n_class, in_channels, out_channels, edge_poolings, batchnorm=nn.BatchNorm2d):
        super(MultiTaskNet, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.edge_pooling1 = nn.AvgPool2d(edge_poolings[0], stride=1, padding=(edge_poolings[0] - 1) // 2)
        self.edge_pooling2 = nn.AvgPool2d(edge_poolings[1], stride=1, padding=(edge_poolings[1] - 1) // 2)
        self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1, 1, bias=False),
                                       nn.ReLU(True),
                                       nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
                                       batchnorm(out_channels),
                                       nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.edge_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                 groups=out_channels),
                                       batchnorm(out_channels),
                                       nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.att_edge = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                groups=out_channels),
                                      batchnorm(out_channels),
                                      nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.seg_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                groups=out_channels),
                                      batchnorm(out_channels),
                                      nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.att_seg = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                               groups=out_channels),
                                     batchnorm(out_channels),
                                     nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        self.edge_supervision = nn.Conv2d(out_channels, n_class, 3, padding=1)
        self.seg_supervision = nn.Conv2d(out_channels, n_class, 3, padding=1)

    def forward(self, x):
        x = self.reduce_conv(x)
        edge1 = self.edge_pooling1(x)
        edge2 = self.edge_pooling2(x)
        x = self.fuse_conv(torch.cat([x, edge1, edge2], dim=1))
        edge_x = self.edge_conv(x)
        seg_x = self.seg_conv(x)
        edge_feats = edge_x + (1 - self.sigmoid(self.att_edge(edge_x))) * seg_x
        seg_feats = seg_x + (1 - self.sigmoid(self.att_seg(seg_x))) * edge_x
        edge_sup = self.edge_supervision(edge_feats)
        seg_sup = self.seg_supervision(seg_feats)
        return seg_feats, edge_feats, seg_sup, edge_sup

class FeatureCrossFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureCrossFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.att_conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.att_conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.att_conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.att_conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
    def forward(self, x1, x2, x3, x4):
        size = x1.size()[2:]
        if x2.size()[2:] != size:
            x2 = F.interpolate(x2, size, mode='bilinear', align_corners=True)
        if x3.size()[2:] != size:
            x3 = F.interpolate(x3, size, mode='bilinear', align_corners=True)
        if x4.size()[2:] != size:
            x4 = F.interpolate(x4, size, mode='bilinear', align_corners=True)

        att1 = self.att_conv1(x1)
        att2 = self.att_conv2(x2)
        att3 = self.att_conv3(x3)
        att4 = self.att_conv4(x4)
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x3 = self.conv(x3)
        x4 = self.conv(x4)

        out = x1 + (1 - self.sigmoid(att1)) * (
                self.sigmoid(att2) * x2 + self.sigmoid(att3) * x3 + self.sigmoid(att4) * x4)
        return out

class StackMultiTaskFusion_head(nn.Module):
    def __init__(self, n_class, in_channels, reduce_dim, dim, batchnorm=nn.BatchNorm2d, has_aspp=False, aspp_dim=None):
        super(StackMultiTaskFusion_head, self).__init__()
        # print('aspp: {}'.format(has_aspp))
        self.has_aspp = has_aspp
        if has_aspp:
            self.aspp_combine = nn.Sequential(nn.ReLU(True),
                                              nn.Conv2d(reduce_dim + aspp_dim, reduce_dim + aspp_dim, 3, 1, 1,
                                                        bias=False, groups=reduce_dim + aspp_dim),
                                              batchnorm(reduce_dim + aspp_dim),
                                              nn.Conv2d(reduce_dim + aspp_dim, reduce_dim, 1, bias=False))
        self.mtl4 = MultiTaskNet(n_class, in_channels[3], reduce_dim, [3, 5], batchnorm)
        self.cf4 = FeatureCrossFusion(reduce_dim,dim[3])
        self.up_conv4 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl3 = MultiTaskNet(n_class, in_channels[2], reduce_dim, [3, 5], batchnorm)
        self.cf3 = FeatureCrossFusion(reduce_dim,dim[2])
        self.up_conv3 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl2 = MultiTaskNet(n_class, in_channels[1], reduce_dim, [5, 7], batchnorm)
        self.cf2 = FeatureCrossFusion(reduce_dim, dim[1])
        self.up_conv2 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl1 = MultiTaskNet(n_class, in_channels[0], reduce_dim, [5, 7], batchnorm)
        self.cf1 = FeatureCrossFusion(reduce_dim, dim[0])
        self.up_conv1 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.conv1 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, 3, 1, 1),
                                   batchnorm(reduce_dim),
                                   nn.ReLU(True))
        self.conv2 = nn.Conv2d(reduce_dim, n_class, 1)

    def forward(self, x1, x2, x3, x4, aspp=None):
        x1_size = x1.size()[2:]
        mtl4_1, mtl4_2, mtl4_3, mtl4_4 = self.mtl4(x4)
        mtl3_1, mtl3_2, mtl3_3, mtl3_4 = self.mtl3(x3)
        mtl2_1, mtl2_2, mtl2_3, mtl2_4 = self.mtl2(x2)
        mtl1_1, mtl1_2, mtl1_3, mtl1_4 = self.mtl1(x1)
        d4 = self.cf4(mtl4_1, mtl1_1, mtl2_1, mtl3_1)
        if aspp is not None:
            d4 = self.aspp_combine(torch.cat((d4, aspp), dim=1))

        d3 = self.cf3(mtl3_1, mtl1_1, mtl2_1, mtl4_1)
        # d3 = self.up_conv3(torch.cat((d3, d4), dim=1))
        d2 = self.cf2(mtl2_1, mtl1_1, mtl3_1, mtl4_1)
        # d2 = self.up_conv2(torch.cat((d2, d3), dim=1))
        d1 = self.cf1(mtl1_1, mtl2_1, mtl3_1, mtl4_1)
        # up_d2 = F.interpolate(d2, x1_size, mode='bilinear', align_corners=True)
        # d1 = self.up_conv1(torch.cat((d1, up_d2), dim=1))
        # out = self.conv1(d1)
        # out = self.conv2(out)
        # return out, mtl1_3, mtl1_4, mtl2_3, mtl2_4, mtl3_3, mtl3_4, mtl4_3, mtl4_4

        return d1, d2, d3, d4

class ChannelMeanMaxAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanMaxAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1_mean = self.relu(self.fc1(squeeze_tensor_mean))
        fc_out_2_mean = self.fc2(fc_out_1_mean)

        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]
        fc_out_1_max = self.relu(self.fc1(squeeze_tensor_max))
        fc_out_2_max = self.fc2(fc_out_1_max)

        a, b = squeeze_tensor_mean.size()
        result = torch.Tensor(a, b)
        result = torch.add(fc_out_2_mean, fc_out_2_max)
        fc_out_2 = F.sigmoid(result)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * input_tensor

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=516, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.conv1(out)

        return out

#########################################################################################
def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)

def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity  # .expand_as()将前面的维度扩为后面的维度
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)

        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out

class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out

class CSNet(nn.Module):
    def __init__(self, classes, channels, aspp_out_dim=256, reduce_dim=128):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet, self).__init__()
        self.enc_input = ResEncoder(channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention(512)
        self.attention_fuse = nn.Conv2d(512 * 2, 512, kernel_size=1)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv2d(32, classes, kernel_size=1)
        initialize_weights(self)

        self.block_channels = [32, 64, 128, 256]
        self.dim = [32, 64, 128, 256]
        self.head = StackMultiTaskFusion_head(classes, self.block_channels, reduce_dim,self.dim, has_aspp=True,
                                              aspp_dim=aspp_out_dim)

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        self.channelmeanmaxattention1 = ChannelMeanMaxAttention(512)
        self.channelmeanmaxattention2 = ChannelMeanMaxAttention(256)
        self.channelmeanmaxattention3 = ChannelMeanMaxAttention(128)
        self.channelmeanmaxattention4 = ChannelMeanMaxAttention(64)
        self.spatialattention = SpatialAttention()


    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        ###############
        d1, d2, d3, d4 = self.head(enc_input, enc1, enc2, enc3)
        ###############

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)
        feater = self.dblock(input_feature)
        feater = self.spp(feater)
        attention_fuse = feater + attention

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((d4, up4), dim=1)
        up4 = self.channelmeanmaxattention1(up4)
        up4 = self.spatialattention(up4)
        dec4 = self.decoder4(up4)


        up3 = self.deconv3(dec4)
        up3 = torch.cat((d3, up3), dim=1)
        up3 = self.channelmeanmaxattention2(up3)
        up3 = self.spatialattention(up3)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((d2, up2), dim=1)
        up2 = self.channelmeanmaxattention3(up2)
        up2 = self.spatialattention(up2)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((d1, up1), dim=1)
        up1 = self.channelmeanmaxattention4(up1)
        up1 = self.spatialattention(up1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final1 = F.sigmoid(final)
        return final1

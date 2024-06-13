import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Util import calc_Eucli_dist_matrix, calc_center_Eucli_dist_matrix, calc_cov_matrix_torch, TSNE_Visualize


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128, k=20, center_dim=128, is_object=False):
        super(ReconstructiveSubNetwork, self).__init__()
        # 中心边界阈值
        self.register_buffer("Tc", torch.zeros(1, 256))
        # 记忆库
        self.register_buffer("memory", torch.zeros(1, center_dim, 16, 16))
        # 聚类中心，可学习参数
        center = torch.zeros(center_dim, k)  # torch.Size([128, 20])
        # nn.init.kaiming_uniform_(center)
        nn.init.kaiming_normal_(center)
        self.center = nn.Parameter(center)  # torch.Size([128, 20])
        # 是否为物体
        self.is_object = is_object

        self.encoder = EncoderReconstructive(in_channels, base_width, center_dim)
        self.cluster = DeepEmbeddingCluster(k)
        self.edit = FineGrainedFeatureEditing()
        self.decoder = DecoderReconstructive(center_dim, base_width, out_channels=out_channels)

    def forward(self, x, is_normal, epoch):
        print("Tc", self.Tc, self.Tc.dtype)
        print("start center nan?", torch.isnan(self.center).any())
        print("encoder has nan?",self.encoder_check())
        f, skip_layer = self.encoder(x) # 编码器
        if is_normal:
            loss = {'Lc': torch.zeros(1).cuda(), 'kl_loss': torch.zeros(1).cuda(),
                    'entropy_loss': torch.zeros(1).cuda(), 'limit_entropy_loss': torch.zeros(1).cuda()}
            if epoch > 50:
                # 1. 正常分支
                # f torch.Size([8, 128, 16, 16])
                self.memory = torch.mean(f, dim=0, keepdim=True)
                f, Tc, loss = self.cluster(f, self.center) # 深度嵌入聚类
                self.Tc = Tc # 中心边界阈值
            output = self.decoder(f, skip_layer)  # 解码器
            return output, loss
        else:
            # 2. 异常分支
            f, Ld = self.edit(f, self.center, self.Tc, self.memory, self.is_object)
            output = self.decoder(f, skip_layer)  # 解码器
            return output, Ld

    def encoder_check(self):
        for params in self.encoder.parameters():
            if torch.isnan(params).any():
                return True
        return False


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.feature_enhance = FeatureEnhance()
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        print("b1-b6 nan?",torch.isnan(b1).any(),torch.isnan(b2).any(),torch.isnan(b3).any(),torch.isnan(b4).any(),torch.isnan(b5).any(),torch.isnan(b6).any())
        f_list = (b1, b2, b3, b4, b5, b6)
        b1, b2, b3, b4, b5, b6 = self.feature_enhance(f_list)
        out_b1, out_b2, out_b3, out_b4, out_b5, out_b6, out_final = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        print("预测图1-7 nan?",torch.isnan(out_b1).any(),torch.isnan(out_b2).any(),torch.isnan(out_b3).any(),torch.isnan(out_b4).any(),torch.isnan(out_b5).any(),torch.isnan(out_b6).any(),torch.isnan(out_final).any())
        if self.out_features:
            return out_b1, out_b2, out_b3, out_b4, out_b5, out_b6, out_final, b2, b3, b4, b5, b6
        else:
            return out_b1, out_b2, out_b3, out_b4, out_b5, out_b6, out_final

class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU())
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU())
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU())
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU())
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU())

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU())


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1,b2,b3,b4,b5,b6

class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        # --自深层至浅层--
        self.up_b6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1), # 512 512
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.fuse_up_b6_b5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1), # 512 512
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )


        self.up_b5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1), # 512 512
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))

        self.fuse_up_b5_b4 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.up_b4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),  # 512 256
                                   nn.BatchNorm2d(base_width * 4),
                                   nn.ReLU(inplace=True))

        self.fuse_up_b4_b3 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up_b3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),  # 256 128
                                   nn.BatchNorm2d(base_width * 2),
                                   nn.ReLU(inplace=True))

        self.fuse_up_b3_b2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up_b2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),  # 128 64
                                   nn.BatchNorm2d(base_width),
                                   nn.ReLU(inplace=True))

        self.fuse_up_b2_b1 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 64 64
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 64 64
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        # --自浅层至深层--
        self.down_b1 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),  # 64 128
                                   nn.BatchNorm2d(base_width * 2),
                                   nn.ReLU(inplace=True))

        self.fuse_down_b1_b2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.down_b2 = nn.Sequential(nn.MaxPool2d(2),
                                     nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),  # 128 256
                                     nn.BatchNorm2d(base_width * 4),
                                     nn.ReLU(inplace=True))
        self.fuse_down_b2_b3 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.down_b3 = nn.Sequential(nn.MaxPool2d(2),
                                     nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),  # 256 512
                                     nn.BatchNorm2d(base_width * 8),
                                     nn.ReLU(inplace=True))
        self.fuse_down_b3_b4 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.down_b4 = nn.Sequential(nn.MaxPool2d(2),
                                     nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
                                     nn.BatchNorm2d(base_width * 8),
                                     nn.ReLU(inplace=True))
        self.fuse_down_b4_b5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.down_b5 = nn.Sequential(nn.MaxPool2d(2),
                                     nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
                                     nn.BatchNorm2d(base_width * 8),
                                     nn.ReLU(inplace=True))
        self.fuse_down_b5_b6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        # --合并两分支--
        self.fuse_b1_merge = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 64 64
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 64 64
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fuse_b2_merge = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),  # 128 128
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.fuse_b3_merge = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),  # 256 256
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.fuse_b4_merge = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.fuse_b5_merge = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.fuse_b6_merge = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 512 512
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        # b1_merge获得预测图
        self.pre_out1 = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=1), # 64 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # b2_merge获得预测图
        self.pre_out2 = nn.Sequential(
            nn.Conv2d(base_width * 2, out_channels, kernel_size=1), # 128 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 128*128  256*256
        )

        # b3_merge获得预测图
        self.pre_out3 = nn.Sequential(
            nn.Conv2d(base_width * 4, out_channels, kernel_size=1), # 256 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 64*64 256*256
        )

        # b4_merge获得预测图
        self.pre_out4 = nn.Sequential(
            nn.Conv2d(base_width * 8, out_channels, kernel_size=1), # 512 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # 32*32 256*256
        )

        # b5_merge获得预测图
        self.pre_out5 = nn.Sequential(
            nn.Conv2d(base_width * 8, out_channels, kernel_size=1), # 512 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 16*16 256*256
        )

        # b6_merge获得预测图
        self.pre_out6 = nn.Sequential(
            nn.Conv2d(base_width * 8, out_channels, kernel_size=1),  # 512 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # 8*8 256*256
        )

        # 拼接预测图生成最终预测图
        self.pre_final = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels, kernel_size=1),  # 12 2
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1,b2,b3,b4,b5,b6):
        # 深层->浅层分支
        up_b6 = self.up_b6(b6)
        b5_out = b5 + up_b6
        b5_out = self.fuse_up_b6_b5(b5_out)

        up_b5 = self.up_b5(b5_out)
        b4_out = b4 + up_b5
        b4_out = self.fuse_up_b5_b4(b4_out)

        up_b4 = self.up_b4(b4_out)
        b3_out = b3 + up_b4
        b3_out = self.fuse_up_b4_b3(b3_out)

        up_b3 = self.up_b3(b3_out)
        b2_out = b2 + up_b3
        b2_out = self.fuse_up_b3_b2(b2_out)

        up_b2 = self.up_b2(b2_out)
        b1_out = b1 + up_b2
        b1_out = self.fuse_up_b2_b1(b1_out)

        # 浅层->深层分支
        down_b1 = self.down_b1(b1)
        b2_out_down = down_b1 + b2
        b2_out_down = self.fuse_down_b1_b2(b2_out_down)

        down_b2 = self.down_b2(b2_out_down)
        b3_out_down = down_b2 + b3
        b3_out_down = self.fuse_down_b2_b3(b3_out_down)

        down_b3 = self.down_b3(b3_out_down)
        b4_out_down = down_b3 + b4
        b4_out_down = self.fuse_down_b3_b4(b4_out_down)

        down_b4 = self.down_b4(b4_out_down)
        b5_out_down = down_b4 + b5
        b5_out_down = self.fuse_down_b4_b5(b5_out_down)

        down_b5 = self.down_b5(b5_out_down)
        b6_out_down = down_b5 + b6
        b6_out_down = self.fuse_down_b5_b6(b6_out_down)

        # 两分支合并
        b1_merge = b1 + b1_out
        b2_merge = b2_out_down + b2_out
        b3_merge = b3_out_down + b3_out
        b4_merge = b4_out_down + b4_out
        b5_merge = b5_out_down + b5_out
        b6_merge = b6_out_down + b6

        b1_merge = self.fuse_b1_merge(b1_merge)
        b2_merge = self.fuse_b2_merge(b2_merge)
        b3_merge = self.fuse_b3_merge(b3_merge)
        b4_merge = self.fuse_b4_merge(b4_merge)
        b5_merge = self.fuse_b5_merge(b5_merge)
        b6_merge = self.fuse_b6_merge(b6_merge)

        # 获得每层预测图
        out_b1 = self.pre_out1(b1_merge)
        out_b2 = self.pre_out2(b2_merge)
        out_b3 = self.pre_out3(b3_merge)
        out_b4 = self.pre_out4(b4_merge)
        out_b5 = self.pre_out5(b5_merge)
        out_b6 = self.pre_out6(b6_merge)

        # 获得最终预测图
        out_cat = torch.cat([out_b1, out_b2, out_b3, out_b4, out_b5, out_b6], dim=1)
        out_final = self.pre_final(out_cat)

        return (out_b1, out_b2, out_b3, out_b4, out_b5, out_b6, out_final)

class FeatureEnhance(nn.Module):
    def __init__(self):
        super(FeatureEnhance, self).__init__()
        self.cjam_b1 = CovarianceJointAttentionModule(channel_dim=64)
        self.cjam_b2 = CovarianceJointAttentionModule(channel_dim=128)
        self.cjam_b3 = CovarianceJointAttentionModule(channel_dim=256)
        self.cjam_b4 = CovarianceJointAttentionModule(channel_dim=512)
        self.cjam_b5 = CovarianceJointAttentionModule(channel_dim=512)
        self.cjam_b6 = CovarianceJointAttentionModule(channel_dim=512)
    def forward(self, f_list):
        b1, b2, b3, b4, b5, b6 = f_list
        b1 = self.cjam_b1(b1)
        b2 = self.cjam_b2(b2)
        b3 = self.cjam_b3(b3)
        b4 = self.cjam_b4(b4)
        b5 = self.cjam_b5(b5)
        b6 = self.cjam_b6(b6)
        return (b1, b2, b3, b4, b5, b6)



class CovarianceJointAttentionModule(nn.Module):
    def __init__(self, channel_dim, reduction=8):
        super(CovarianceJointAttentionModule, self).__init__()

        self.SA = Spatial_attention(kernel_size=3) # 空间注意力
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.cos_sim = nn.CosineSimilarity(dim=1) # 余弦相似度
        self.reduction = reduction

        self.fuse_block = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(),
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(),
        )

        # 处理GCP的MLP
        self.mlp_GCP = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel_dim, channel_dim // self.reduction, 1, bias=False),
            # inplace=False直接替换，节省内存
            nn.ReLU(inplace=False),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel_dim // self.reduction, channel_dim, 1, bias=False)
        )

        # 处理协方差矩阵的MLP
        self.mlp_cov = nn.Sequential(
            # Conv2d比Linear方便操作
            nn.Linear(channel_dim, channel_dim // reduction, bias=False),
            # nn.Conv2d(channel_dim, channel_dim // 16, 1, bias=False),
            # inplace=False直接替换，节省内存
            nn.ReLU(inplace=False),
            nn.Linear(channel_dim // reduction, channel_dim, bias=False),
            # nn.Conv2d(channel_dim // 16, channel_dim, 1, bias=False)
        )

    def forward(self, f): # torch.Size([8, 512, 16, 16])

        # 计算协方差矩阵
        cov_matrix = calc_cov_matrix_torch(f) # torch.Size([8, 512, 512])

        # --局部分支--
        # 计算协方差通道注意力权重
        cov_pool = torch.mean(cov_matrix, dim=2, keepdim=True).unsqueeze(-1)  # torch.Size([8, 512, 1, 1])
        cov_pool = self.mlp_GCP(cov_pool)  # 512 -> 64 -> 512
        channel_weight = torch.sigmoid(cov_pool).squeeze(-1)  # torch.Size([8, 512, 1])

        # 计算空间注意力Map
        f_sa = self.SA(f) # torch.Size([8, 512, 16, 16])
        max_pool = self.max_pool(f_sa) # torch.Size([8, 512, 1, 1])
        sim_matrix = self.cos_sim(max_pool, f_sa).unsqueeze(1) # torch.Size([8, 1, 16, 16])
        sim_weight = torch.sigmoid(sim_matrix)  # torch.Size([8, 1, 16, 16])
        b, c, h, w = sim_weight.shape
        sim_weight = sim_weight.view(b, c, -1) # torch.Size([8, 1, 256])

        # 计算通道-空间联合坐标注意力  8, 512, 1   8, 1, 256
        coor_weight = torch.bmm(channel_weight, sim_weight) # torch.Size([8, 512, 256])
        b, c, _ = coor_weight.shape # 8 512 256
        coor_weight = coor_weight.view(b, c, h, w) # torch.Size([8, 512, 16, 16])

        # 增强原始特征
        f_local = f * coor_weight # torch.Size([8, 512, 16, 16])

        # --全局分支--
        # 计算协方差相关性权重，建模通道维度长程依赖关系
        cov_matrix = self.mlp_cov(cov_matrix) # 512 -> 64 -> 512
        cov_weight = torch.sigmoid(cov_matrix) # torch.Size([8, 512, 512])
        b, c, h, w = f.shape # 8, 512, 16, 16
        f_ = f.view(b, c, -1) # torch.Size([8, 512, 256])

        # 捕获通道维度全局特征信息 8, 512, 512   8, 512, 256
        f_global = torch.bmm(cov_weight, f_) # torch.Size([8, 512, 256])
        f_global = f_global.view(b, c, h, w) # torch.Size([8, 512, 16, 16])

        # 两分支特征融合
        f_fuse = f_local + f_global
        f_fuse = self.fuse_block(f_fuse)

        return f_fuse


# 空间注意力机制
class Spatial_attention(nn.Module):
    # 初始化，卷积核大小为3*3
    def __init__(self, kernel_size=3):
        # 继承父类初始化方法
        super(Spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 3*3卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        return outputs

if __name__ == '__main__':
    cjam = CovarianceJointAttentionModule(channel_dim=512).cuda()
    x = torch.rand([8, 512, 16, 16]).cuda()
    cjam(x)

class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width, center_dim):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU())
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU())
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU())
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU())
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU())

        # bottleneck
        self.conv1 = nn.Sequential(
            nn.Conv2d(base_width * 8, center_dim, kernel_size=1),
            nn.BatchNorm2d(center_dim),
            nn.ReLU(),
        )


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)

        # 1*1卷积压缩通道 (bottleneck)
        f = self.conv1(b5)  # f torch.Size([8, 128, 16, 16])

        skip_layer = {'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4}
        return f, skip_layer


class DecoderReconstructive(nn.Module):
    def __init__(self, center_dim, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        # 复原通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(center_dim, base_width * 8, kernel_size=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, f, skip_layer):
        b1 = skip_layer['b1']
        b2 = skip_layer['b2']
        b3 = skip_layer['b3']
        b4 = skip_layer['b4']

        # 1*1卷积复原通道
        b5 = self.conv1(f)

        up1 = self.up1(b5)
        up1 = up1 + b4
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        # up2 = up2 + b3
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        # up3 = up3 + b2
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        # up4 = up4 + b1
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out

class DeepEmbeddingCluster(nn.Module):
    def __init__(self, k=20):
        super(DeepEmbeddingCluster, self).__init__()
        # 平滑因子
        # self.fac = nn.Parameter(torch.ones(1,k))


    def forward(self, f, center): # f torch.Size([8, 128, 16, 16])
        print()
        # 依次计算f中每个特征和所有center之间的欧式距离
        dist_matrix = calc_Eucli_dist_matrix(f, center) # torch.Size([8, 256, 20])
        print("normal f nan?",torch.isnan(f).any())
        print("center nan?",torch.isnan(center).any())
        print("dist_matrix nan?", torch.isnan(dist_matrix).any())

        # 计算源分布
        # S_temp = -(self.fac.unsqueeze(0) * dist_matrix)
        S_temp = -(dist_matrix)
        S = torch.softmax(S_temp, dim=2)
        print("源分布S nan?",torch.isnan(S).any())

        # # 计算目标分布
        # tk = torch.sum(S, dim=1, keepdim=True)
        # T_temp = torch.pow(S, 2) / tk
        # T = T_temp / T_temp.norm(p=1, dim=2, keepdim=True)
        # print("目标分布T nan?", torch.isnan(T).any())

        # 计算目标分布
        T_temp = torch.pow(S, 2)
        T = T_temp / T_temp.norm(p=1, dim=2, keepdim=True)
        print("目标分布T nan?", torch.isnan(T).any())

        # 依次计算f中每个特征到最近center的距离
        d, idx = torch.min(dist_matrix, dim=2)
        print("normal d nan?",torch.isnan(d).any())

        # 计算中心边界
        # Tc = torch.mean(d) + torch.std(d)
        Tc = torch.mean(d, dim=0, keepdim=True) + torch.std(d, dim=0, keepdim=True, unbiased=False)

        # 计算两两聚类中心之间的距离
        center_dist = calc_center_Eucli_dist_matrix(center)
        print("center_dist nan?",torch.isnan(center_dist).any())

        # 中心约束损失
        Lc = torch.mean(d) / torch.mean(center_dist)
        print('Lc', Lc, torch.mean(d), torch.mean(center_dist))


        # KL散度损失
        kl_loss = F.kl_div(S.log(), T, reduction='batchmean')
        print("kl_loss", kl_loss)
        print("源分布", torch.max(S,dim=2)[1], "目标分布", torch.max(T,dim=2)[1])

        # 熵损失
        mask = (S == 0).float()
        masked_S = S + mask
        entropy_loss = -S * torch.log(masked_S)
        b, _, _ = entropy_loss.shape
        entropy_loss = entropy_loss.sum() / b
        print("entropy_loss", entropy_loss)

        # 避免大多数实例被分配给同一个集群
        # torch.Size([8, 256, 20])
        limit_S = torch.sum(S, dim=1, keepdim=True) # torch.Size([8, 1, 20])
        limit_S = limit_S / limit_S.norm(p=1, dim=2, keepdim=True) # torch.Size([8, 1, 20])
        mask = (limit_S == 0).float()
        masked_S = limit_S + mask
        limit_entropy_loss = -limit_S * torch.log(masked_S)
        b, _, _ = limit_entropy_loss.shape
        limit_entropy_loss = limit_entropy_loss.sum() / b


        # loss = {'Lc': Lc, 'kl_loss': kl_loss, 'entropy_loss': entropy_loss}
        loss = {'Lc': Lc, 'kl_loss': kl_loss, 'entropy_loss': entropy_loss, 'limit_entropy_loss': limit_entropy_loss}

        return f, Tc, loss


class FineGrainedFeatureEditing(nn.Module):
    def __init__(self):
        super(FineGrainedFeatureEditing, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=2)
        self.relu = nn.ReLU()
        self.epsilon = 1e-10

    def forward(self, f, center, Tc, memory, is_object): # f torch.Size([8, 128, 16, 16])

        # 依次计算f中每个特征和所有center之间的欧式距离
        dist_matrix = calc_Eucli_dist_matrix(f, center)  # torch.Size([8, 256, 20])
        print("anomaly f nan?", torch.isnan(f).any())
        print("center nan?", torch.isnan(center).any())
        print("dist_matrix nan?", torch.isnan(dist_matrix).any())

        # 依次计算f中每个特征到最近center的距离
        d, idx = torch.min(dist_matrix, dim=2) # torch.Size([8, 256])
        print("anomaly d nan?", torch.isnan(d).any())

        # #-----
        # S_temp = -(self.fac.unsqueeze(0) * dist_matrix)
        # S = torch.softmax(S_temp, dim=2)
        # idx = torch.max(S, dim=2)[1]
        # print(idx)
        # b,c,h,w = f.shape # 1 128 16 16
        # f_v = f.view(b,c,-1).squeeze(0).permute(1,0) # 256 128
        # center_v = center.permute(1, 0) # 20 128
        # TSNE_Visualize(f_v, idx, center_v)
        # #-----

        # 检测距离大于Tc的异常特征
        idx_ano = d > Tc
        idx_nor = torch.logical_not(idx_ano)

        b, c, h, w = f.shape  # f torch.Size([8, 128, 16, 16])
        n = h * w
        f = f.view(b, c, n).permute(0, 2, 1)  # 8 256 128

        if is_object:
            print("物体编辑")
            # 物体类别
            # 将异常特征和正常特征筛选出来
            m_b, _, _, _ = memory.shape # 1, 128, 16, 16
            memory = memory.view(m_b, c, n).permute(0, 2, 1) # 1 256 128

            # for i in range(b):
            #     f[i:i+1,...][idx_ano[i:i+1,...]] = memory[idx_ano[i:i+1,...]]

            memory = memory.expand(b, n, c) # 8 256 128
            f[idx_ano] = memory[idx_ano]

            f = f.permute(0, 2, 1).view(b, c, h, w)  # 8 128 16 16
        else:
            print("纹理编辑")
            # 纹理类别
            # 将异常特征和正常特征筛选出来
            ano_f = f[idx_ano] # torch.Size([65, 128])
            nor_f = f[idx_nor] # torch.Size([1983, 128])

            # 依次计算每个异常特征和正常特征之间的余弦相似度
            m, _ = ano_f.shape # 65
            t, _ = nor_f.shape # 1983
            ano_f = ano_f.unsqueeze(1).expand(m, t, c) # torch.Size([65, 1983, 128])
            nor_f = nor_f.unsqueeze(0).expand(m, t, c) # torch.Size([65, 1983, 128])
            sim_matrix = self.cos_sim(ano_f, nor_f) # torch.Size([65, 1983])
            sim_weight = torch.softmax(sim_matrix, dim=1) # torch.Size([65, 1983])
            print("sim_weight nan?",torch.isnan(sim_weight).any(), "sim_weight shape", sim_weight.shape)

            # 根据阈值，去掉小的，保留大的，再归一化
            thr = torch.mean(sim_weight, dim=1, keepdim=True) - 2 * torch.std(sim_weight, dim=1, keepdim=True, unbiased=False)
            print("thr nan?", torch.isnan(thr).any(), torch.isnan(torch.mean(sim_weight, dim=1, keepdim=True)).any(),torch.isnan(torch.std(sim_weight, dim=1, keepdim=True)).any())
            sim_weight = (self.relu(sim_weight - thr) * sim_weight) / (torch.abs(sim_weight - thr) + self.epsilon)
            print("去小sim_weight nan?", torch.isnan(sim_weight).any())
            # sim_weight = sim_weight / sim_weight.norm(p=1, dim=1, keepdim=True) # torch.Size([65, 1983])
            sim_weight = sim_weight / torch.sum(sim_weight, dim=1, keepdim=True) # torch.Size([65, 1983])
            print("去小归一sim_weight nan?",torch.isnan(sim_weight).any(), torch.isnan(torch.sum(sim_weight, dim=1, keepdim=True)).any())

            # 重新编辑异常特征 65 1983 * 1983 128 -> 65 128
            f[idx_ano] = torch.mm(sim_weight, f[idx_nor])
            print("编辑后的异常f nan?",torch.isnan(f[idx_ano]).any())
            f = f.permute(0, 2, 1).view(b, c, h, w) # 8 128 16 16
            print("编辑后的f nan?",torch.isnan(f).any())

        # 类间差异损失
        # d_weight = torch.softmax(d, dim=1)
        d_weight = d
        ano_weight = d_weight[idx_ano]
        nor_weight = d_weight[idx_nor]
        print("大的",ano_weight.shape,"小的",nor_weight.shape)

        if ano_weight.shape[0] > 0:
            Ld = torch.mean(nor_weight) / (torch.mean(ano_weight) + torch.tensor(1e-3))
        else:
            Ld = torch.mean(nor_weight)
        print('Ld', Ld, torch.mean(nor_weight), (torch.mean(ano_weight) + torch.tensor(1e-3)), ano_weight.shape)
        return f, Ld
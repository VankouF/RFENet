import torch
import torch.nn as nn
import torch.nn.functional as F

from network import resnet_d as Resnet_Deep
from network.resnext import resnext101_32x8
from network.nn.mynn import Norm2d
from network.nn.operators import _AtrousSpatialPyramidPoolingModule
from network.mutual_process.edge_point_process import RegionPointProcess


class RFENet(nn.Module):

    def __init__(self, args, num_classes, criterion=None, trunk='seresnext-50', variant='D',
                 skip='m1', skip_num=256):
        super(RFENet, self).__init__()
        self.args = args
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_mum = skip_num
        self.num_cascade = args.num_cascade

        self.edge_map_iteration = None
        self.region_map_iteration = None

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101-32x8':
            resnet = resnext101_32x8()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError('Only support resnet50 and resnet101 for now')

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print('Not using dilation')

        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=2048, reduction_dim=256,
                                                       output_stride=8 if self.variant == 'D' else 16)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.aspp_fine = nn.Sequential(
            nn.Conv2d(256, skip_num, kernel_size=3, padding=1, bias=False),
            Norm2d(skip_num),
            nn.ReLU(inplace=True))

        if self.skip == 'm1':
            self.bot_fine = nn.ModuleList([nn.Sequential(
                nn.Conv2d(256, skip_num, kernel_size=3, padding=1, bias=False),
                Norm2d(skip_num),
                nn.ReLU(inplace=True)) for _ in range(self.num_cascade)])
        elif self.skip == 'm2':
            self.bot_fine = nn.ModuleList([nn.Sequential(
                nn.Conv2d(512, skip_num, kernel_size=3, padding=1, bias=False),
                Norm2d(skip_num),
                nn.ReLU(inplace=True)) for _ in range(self.num_cascade)])
        else:
            raise ValueError('Not a valid skip')

        self.layer_feat_fine = nn.ModuleList()
        for i in range(self.num_cascade - 1):
            inchannels = 2 ** (-i + 11)
            self.layer_feat_fine.append(nn.Sequential(
                nn.Conv2d(inchannels, skip_num, kernel_size=3, padding=1, bias=False),
                Norm2d(skip_num),
                nn.ReLU(inplace=True)))

        self.merge_feat = [nn.Sequential(
            nn.Conv2d(256 + skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
            for _ in range(self.num_cascade - 1)]
        self.merge_feat = nn.ModuleList(self.merge_feat)

        self.merge_boundary = [BoundaryExtractor(low_channels=skip_num, high_channels=256)
                               for _ in range(self.num_cascade)]

        self.merge_boundary = nn.ModuleList(self.merge_boundary)

        self.mutualfus = [MutualFusion(inplane_x=256, inplane_b=256)
                          for _ in range(self.num_cascade)]
        self.mutualfus = nn.ModuleList(self.mutualfus)

        self.edge_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade - 1)]
        self.edge_out_pre = nn.ModuleList(self.edge_out_pre)

        self.final_seg_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade - 1)]
        self.final_seg_out_pre = nn.ModuleList(self.final_seg_out_pre)

        self.edge_out_refine = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_refine = nn.ModuleList(self.edge_out_refine)

        self.seg_out_refine = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.seg_out_refine = nn.ModuleList(self.seg_out_refine)

        self.region_process = nn.ModuleList(
            [RegionPointProcess(args=args,
                                dim=256,
                                region_num_points=args.region_num_points,
                                edge_num_points=args.edge_num_points,
                                num_heads=args.num_heads,
                                mlp_ratio=args.mlp_ratio,

                                )
             for _ in range(self.num_cascade)
             ])

        self.semantic_predict = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1, bias=True)) for _ in range(self.num_cascade)
            ])

        self.edge_predict = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1, bias=True)) for _ in range(self.num_cascade)
            ])

        self.semantic_fine = nn.Sequential(
            nn.Conv2d(self.num_cascade * 256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.semantic_final_predict = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

    def forward(self, x, gts=None):

        x_size = x.size()

        feats, seg_edges, seg_finals = [], [], []

        seg_edges_refine, seg_finals_refine = [], []

        feats.append(self.layer0(x))
        feats.append(self.layer1(feats[0]))
        feats.append(self.layer2(feats[1]))
        feats.append(self.layer3(feats[2]))
        feats.append(self.layer4(feats[3]))
        aspp = self.bot_aspp(self.aspp(feats[-1]))

        fine_size = feats[1].size()

        for i in range(self.num_cascade):
            low_feat = self.bot_fine[i](feats[1])
            if i == 0:
                last_seg_feat = F.interpolate(aspp, size=fine_size[2:], mode='bilinear', align_corners=True)

                aspp_up = F.interpolate(self.aspp_fine(aspp), size=fine_size[2:], mode='bilinear', align_corners=True)
                last_boundary_feat = self.merge_boundary[i](aspp_up, low_feat)
            else:
                last_seg_feat = seg_finals[-1]
                last_boundary_feat = seg_edges[-1]

                layer_feat = F.interpolate(self.layer_feat_fine[i - 1](feats[-i]), size=fine_size[2:], mode='bilinear',
                                           align_corners=True)
                last_seg_feat = self.merge_feat[-i](torch.cat([last_seg_feat, layer_feat], dim=1))

                last_boundary_feat = self.merge_boundary[i](last_boundary_feat, low_feat)

            last_seg_feat, last_boundary_feat = self.mutualfus[i](last_seg_feat, last_boundary_feat)

            if i != self.num_cascade - 1:
                seg_edge_pre = self.edge_out_pre[i](last_boundary_feat)
                seg_final_pre = self.final_seg_out_pre[i](last_seg_feat)

                seg_edges.append(seg_edge_pre)
                seg_finals.append(seg_final_pre)

            seg_edges_refine.append(self.edge_out_refine[i](last_boundary_feat))
            seg_finals_refine.append(self.seg_out_refine[i](last_seg_feat))

        edge_predictions = [edge_predict(edge_refine) for edge_refine, edge_predict in
                            zip(seg_edges_refine, self.edge_predict)]
        semantic_predictions = [semantic_predict(semantic_refine) for semantic_refine, semantic_predict in
                                zip(seg_finals_refine, self.semantic_predict)]

        semantic_refines = [region_process(seg_finals_refine[i], semantic_predictions[i].clone().detach(),
                                           torch.sigmoid(edge_predictions[i].clone().detach()))
                            for i, region_process in enumerate(self.region_process)]

        semantic_predictions = [semantic_predict(semantic_refine) for semantic_refine, semantic_predict in
                                zip(semantic_refines, self.semantic_predict)]

        semantic_fine = self.semantic_fine(torch.cat(semantic_refines, dim=1))

        semantic_final_prediction = self.semantic_final_predict(semantic_fine)

        edge_predictions = [F.interpolate(edge_prediction, size=x_size[2:], mode='bilinear', align_corners=True) for
                            edge_prediction in edge_predictions]
        semantic_predictions = [F.interpolate(semantic_prediction, size=x_size[2:], mode='bilinear', align_corners=True)
                                for semantic_prediction in semantic_predictions]
        semantic_final_prediction = F.interpolate(semantic_final_prediction, size=x_size[2:], mode='bilinear',
                                                  align_corners=True)

        if self.training:
            return self.criterion((semantic_final_prediction, semantic_predictions, edge_predictions), gts)

        return semantic_final_prediction

    def get_edge_map_per_iteration(self):
        return self.edge_map_iteration

    def get_region_map_per_iteration(self):
        return self.region_map_iteration


class MutualFusion(nn.Module):

    def __init__(self, inplane_x, inplane_b, norm_layer=Norm2d, dr2=2, dr4=4):
        super(MutualFusion, self).__init__()

        self.inplane_x = inplane_x
        self.inplane_b = inplane_b

        self.input_channels = inplane_x + inplane_b

        self.channels_single = int(self.input_channels / 4)
        self.channels_double = int(self.input_channels / 2)

        self.dr2 = dr2
        self.dr4 = dr4

        self.padding2 = 2 * dr2

        self.padding4 = 4 * dr4

        self.A = None

        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p2_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, padding=self.padding2,
                      dilation=self.dr2),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1),
                                       norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 9, 1, padding=self.padding4,
                      dilation=self.dr4),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1),
                                       norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_double, out_channels=2, kernel_size=1),
        )

    def forward(self, x, x_b):
        concat_feature = torch.cat([x, x_b], dim=1)

        p2_input = self.p2_channel_reduction(concat_feature)
        p2 = self.p2_fusion(self.p2_d1(p2_input))

        p4_input = self.p4_channel_reduction(concat_feature)
        p4 = self.p4_fusion(self.p4_d1(p4_input))

        A = torch.sigmoid(self.channel_reduction(torch.cat((p2, p4), 1)))

        x = x + x * torch.unsqueeze(A[:, 0, :, :], dim=1)
        x_b = x_b + x_b * torch.unsqueeze(A[:, 1, :, :], dim=1)

        self.A = A.data.detach()

        return x, x_b


class BoundaryExtractor(nn.Module):
    def __init__(self, low_channels=256, high_channels=256):
        super(BoundaryExtractor, self).__init__()

        channels = low_channels + high_channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

    def forward(self, high_feat, low_feat):
        x = torch.cat([low_feat, high_feat], dim=1)
        return self.block(x)


def RFENet_resnet50_os16(args, num_classes, criterion):
    return RFENet(args, num_classes, criterion, trunk='resnet-50-deep', variant='D16', skip='m1')


def RFENet_resnet50_os8(args, num_classes, criterion):
    return RFENet(args, num_classes, criterion, trunk='resnet-50-deep', variant='D', skip='m1')


def RFENet_resnext101_os16(args, num_classes, criterion):
    return RFENet(args, num_classes, criterion, trunk='resnext-101-32x8', variant='D16', skip='m1')


def RFENet_resnext101_os8(args, num_classes, criterion):
    return RFENet(args, num_classes, criterion, trunk='resnext-101-32x8', variant='D', skip='m1')

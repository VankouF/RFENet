import torch
import torch.nn as nn
from .cross_attention import CrossAttention
import numpy as np
from .norm import trunc_normal_


def select_from_topK(uncertainty_map, num_points):
    R, _, H, W = uncertainty_map.shape

    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    return point_indices


def get_edge_index(args, uncertainty_map, num_points):
    R, _, H, W = uncertainty_map.shape

    point_indices = select_from_topK(uncertainty_map, num_points)

    return point_indices


def get_region_index_entropy(seg_map, num_points):
    '''

    Args:
        seg_map: B*C*H*W， values belong to [0,1]
        num_points:

    Returns:

    '''
    entropy_map = seg_map * torch.log(seg_map)
    entroy_map = -1 * torch.sum(entropy_map, dim=1, keepdim=True)
    R, _, H, W = entroy_map.shape
    point_indices = torch.topk(entroy_map.view(R, H * W), k=num_points, dim=1)[1]
    return point_indices


def point_sample(input, point_indices, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_indices (Tensor): A tensor of shape (N, P) or (N, Hgrid, Wgrid, 2) that contains sampled indices.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    N, C, H, W = input.shape
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    flatten_input = input.flatten(start_dim=2)
    sampled_feats = flatten_input.gather(dim=2, index=point_indices).view_as(point_indices)
    return sampled_feats


class RegionPointProcess(nn.Module):
    def __init__(self, args, dim, region_num_points, edge_num_points, num_heads, mlp_ratio):
        super(RegionPointProcess, self).__init__()

        self.args = args
        self.region_num_points = region_num_points
        self.edge_num_points = edge_num_points
        self.cross_attention = CrossAttention(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, x, x_logits, edge_prediction):
        B, C, H, W = x.size()

        x_predict = torch.softmax(x_logits, dim=1)
        region_index = get_region_index_entropy(x_predict, self.region_num_points)

        edge_index = get_edge_index(self.args, edge_prediction, self.edge_num_points)

        q_features = point_sample(x, region_index).permute(0, 2, 1)
        kv_features = point_sample(x, edge_index).permute(0, 2, 1)

        cross_attention_features = self.cross_attention(q_features, kv_features, kv_features)
        cross_attention_features = cross_attention_features.permute(0, 2, 1)

        region_index = region_index.unsqueeze(1).expand(-1, C, -1)

        final_features = x.reshape(B, C, H * W).scatter(2, region_index, cross_attention_features).view(B, C, H, W)

        return final_features

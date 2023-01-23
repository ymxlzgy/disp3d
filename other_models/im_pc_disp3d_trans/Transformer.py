import torch
import torch.nn as nn

from timm.models.layers import DropPath,trunc_normal_

from .dgcnn_group import DGCNN_Grouper
from .logger import *
import numpy as np
from knn_cuda import KNN
knn = KNN(k=8, transpose_mode=False)

def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)  # bs k np
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    
    return idx  # bs*k*np

def get_graph_feature(x, knn_index, x_q=None):

        #x: bs, np, c, knn_index: bs*k*np
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        feature = feature.view(batch_size, k, num_query, num_dims)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
        feature = torch.cat((feature - x, x), dim=-1)
        return feature  # b k np c

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # with a shape of 64 x 256
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None:
            knn_f = get_graph_feature(norm_x, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_f], dim=-1)
            x_1 = self.merge_map(x_1)
        
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim_img=768, embed_dim_pc=768, depth=[6, 6], num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        num_query_img = 1024, num_query_pc = 224, knn_layer = -1):
        super().__init__()

        self.embed_dim_img = embed_dim_img
        self.embed_dim_pc = embed_dim_pc
        
        self.knn_layer = knn_layer

        print_log(' Transformer with knn_layer %d' % self.knn_layer, logger='MODEL')

        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        # self.pos_embed = nn.Sequential(
        #     nn.Conv1d(in_chans, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(128, embed_dim, 1)
        # )
        # # self.pos_embed_wave = nn.Sequential(
        # #     nn.Conv1d(60, 128, 1),
        # #     nn.BatchNorm1d(128),
        # #     nn.LeakyReLU(negative_slope=0.2),
        # #     nn.Conv1d(128, embed_dim, 1)
        # # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_proj = nn.Sequential(
            nn.Conv1d(256, embed_dim_pc, 1),
            nn.BatchNorm1d(embed_dim_pc),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim_pc, embed_dim_pc, 1)
        )

        self.img_encoder = nn.ModuleList([
            Block(
                dim=embed_dim_img, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])
        self.pc_encoder = nn.ModuleList([
            Block(
                dim=embed_dim_pc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        # self.increase_dim = nn.Sequential(
        #     nn.Linear(embed_dim,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )

        self.increase_dim_pc = nn.Sequential(
            nn.Conv1d(embed_dim_pc, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.increase_dim_img = nn.Sequential(
            nn.Conv1d(embed_dim_img, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query_img = num_query_img
        self.num_query_pc = num_query_pc
        self.coarse_pred_img = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query_img)
        )
        self.coarse_pred_pc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query_pc)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim_pc, 1)
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim_pc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, imgs_feat, pc_feat):
        # build point proxy
        assert imgs_feat.size(0) == pc_feat.size(0)
        bs = imgs_feat.size(0)

        # NOTE: try to use a sin wave  coor B 3 N, change the pos_embed input dim
        """
        pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        # pos =  self.pos_embed(coor).transpose(1,2)
        x = self.input_proj(f).transpose(1,2)
        """
        # NOTE: YIDA modfied following parts using image features
        x = imgs_feat
        coor, f = self.grouper(pc_feat.transpose(1, 2).contiguous())
        knn_index = get_knn_index(coor)
        pos = self.pos_encoding_sin_wave(coor).transpose(1, 2)
        y = self.input_proj(f).transpose(1, 2)

        # NOTE: GY modfied following parts using image+pc features
        for i, blk in enumerate(self.img_encoder):
            x = blk(x)
        for i, blk in enumerate(self.pc_encoder):
            if i < self.knn_layer:
                y = blk(y + pos, knn_index)   # B N C
            else:
                y = blk(y + pos)

        global_feature_img = self.increase_dim_img(x.transpose(1,2)) # B 1024 N1
        global_feature_pc = self.increase_dim_pc(y.transpose(1, 2))  # B 1024 N2

        global_feature = torch.cat((global_feature_img,global_feature_pc),dim=2) # B 1024 N1+N2
        global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024

        coarse_point_cloud = self.coarse_pred_pc(global_feature).reshape(bs, -1, 3)  #  B M C(3)

        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())
        cross_knn_index = get_knn_index(coor_k=coor, coor_q=coarse_point_cloud.transpose(1, 2).contiguous())

        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query_pc, -1),
            coarse_point_cloud], dim=-1) # B M C+3 
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C 
        # decoder
        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, y, new_knn_index, cross_knn_index)   # B M C
            else:
                q = blk(q, y)

        return q, coarse_point_cloud


class PCTransformer_encoder(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """

    def __init__(self, in_chans=3, embed_dim_img=768, embed_dim_pc=768, depth=[6, 6], num_heads=8, mlp_ratio=2.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_query_img=1024, num_query_pc=224, knn_layer=-1):
        super().__init__()

        self.embed_dim_img = embed_dim_img
        self.embed_dim_pc = embed_dim_pc

        self.knn_layer = knn_layer

        print_log(' Transformer with knn_layer %d' % self.knn_layer, logger='MODEL')

        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        # self.pos_embed = nn.Sequential(
        #     nn.Conv1d(in_chans, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(128, embed_dim, 1)
        # )
        # # self.pos_embed_wave = nn.Sequential(
        # #     nn.Conv1d(60, 128, 1),
        # #     nn.BatchNorm1d(128),
        # #     nn.LeakyReLU(negative_slope=0.2),
        # #     nn.Conv1d(128, embed_dim, 1)
        # # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_proj = nn.Sequential(
            nn.Conv1d(256, embed_dim_pc, 1),
            nn.BatchNorm1d(embed_dim_pc),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim_pc, embed_dim_pc, 1)
        )

        self.img_encoder = nn.ModuleList([
            Block(
                dim=embed_dim_img, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])
        self.pc_encoder = nn.ModuleList([
            Block(
                dim=embed_dim_pc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        # self.increase_dim = nn.Sequential(
        #     nn.Linear(embed_dim,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )

        self.increase_dim_pc = nn.Sequential(
            nn.Conv1d(embed_dim_pc, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.increase_dim_img = nn.Sequential(
            nn.Conv1d(embed_dim_img, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query_img = num_query_img
        self.num_query_pc = num_query_pc
        self.coarse_pred_img = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query_img)
        )
        self.coarse_pred_pc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query_pc)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim_pc, 1)
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim_pc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64  #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda()
        freqs = np.pi * (2 ** freqs)

        freqs = freqs.view(*[1] * len(normal_coor.shape), -1)  # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1)  # B x 3 x N x 1
        k = normal_coor * freqs  # B x 3 x N x D
        s = torch.sin(k)  # B x 3 x N x D
        c = torch.cos(k)  # B x 3 x N x D
        x = torch.cat([s, c], -1)  # B x 3 x N x 2D
        pos = x.transpose(-1, -2).reshape(coor.shape[0], -1, coor.shape[-1])  # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, imgs_feat, pc_query):
        # build point proxy
        assert imgs_feat.size(0) == pc_query.size(0)
        bs = imgs_feat.size(0)

        # NOTE: try to use a sin wave  coor B 3 N, change the pos_embed input dim
        """
        pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        # pos =  self.pos_embed(coor).transpose(1,2)
        x = self.input_proj(f).transpose(1,2)
        """
        # NOTE: YIDA modfied following parts using image features
        x = imgs_feat
        y = pc_query

        # NOTE: GY modfied following parts using image+pc features
        for i, blk in enumerate(self.img_encoder):
            x = blk(x)
        for i, blk in enumerate(self.pc_encoder):
            if i < self.knn_layer:
                y = blk(y, knn_index)  # B N C
            else:
                y = blk(y)

        global_feature_img = self.increase_dim_img(x.transpose(1, 2))  # B 1024 N1
        global_feature_pc = self.increase_dim_pc(y.transpose(1, 2))  # B 1024 N2

        global_feature = torch.cat((global_feature_img, global_feature_pc), dim=2)  # B 1024 N1+N2
        global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024

        coarse_point_cloud = self.coarse_pred_pc(global_feature).reshape(bs, -1, 3)  # B M C(3)

        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())
        cross_knn_index = get_knn_index(coor_k=coor, coor_q=coarse_point_cloud.transpose(1, 2).contiguous())

        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query_pc, -1),
            coarse_point_cloud], dim=-1)  # B M C+3
        q = self.mlp_query(query_feature.transpose(1, 2)).transpose(1, 2)  # B M C
        # decoder
        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, y, new_knn_index, cross_knn_index)  # B M C
            else:
                q = blk(q, y)

        return q, coarse_point_cloud
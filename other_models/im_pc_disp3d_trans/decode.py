import torch
import torch.nn as nn
import torch.nn.functional as F
from other_models.im_pc_disp3d_trans.Transformer import DecoderBlock
from other_models.im_pc_disp3d_trans.encode import OperatorND,Operator3D,Pooling
from other_models.displace.tree import TreeGCN

from math import ceil

def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(
        distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


class Discriminator(nn.Module):
    def __init__(self, batch_size, features):
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(
                nn.Conv1d(
                    features[inx], features[inx + 1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(features[-1], features[-1]),
            nn.Linear(features[-1], features[-2]),
            nn.Linear(features[-2], features[-2]), nn.Linear(features[-2], 1))

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out)  # (B, 1)

        return out


class Decoder(nn.Module):
    def __init__(self, features, degrees, support, neighbor_num, root_num, config):
        # self.batch_size = batch_size
        super(Decoder, self).__init__()
        self.refiner = config.refiner
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query_img = config.num_query_img
        self.trans_dim_img = config.trans_dim_img
        self.num_query_pc = config.num_query_pc
        self.trans_dim_pc = config.trans_dim_pc
        self.neighbor_num = neighbor_num

        self.conv_0 = Operator3D(kernel_num=32, support_num=support)
        self.dropping = torch.nn.Dropout(p=0.1, inplace=True)
        self.conv_1 = OperatorND(32, 64, support_num=support)

        self.conv_2 = Operator3D(kernel_num=32, support_num=support)
        self.conv_3 = OperatorND(32, 64, support_num=support)

        self.conv_4 = OperatorND(64, 64, support_num=support)
        self.conv_5 = OperatorND(64, 64, support_num=support)

        self.conv_6 = OperatorND(512, 1024, support_num=support)

        self.upsampling_1 = TreeGCN(0, features, 8, support=support, node=self.num_query_pc,upsample=True, activation=True)
        self.upsampling_2 = TreeGCN(0, features, 8, support=support, node=self.num_query_pc*8, upsample=True, activation=False)

        self.layer_num = len(features) - 1
        assert self.layer_num == len(
            degrees
        ), "Number of features should be one more than number of degrees."
        self.pointcloud = None


    def forward(self, coase_points):
        neighbor_index = get_neighbor_index(coase_points, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, coase_points)
        fm_0 = self.dropping(fm_0)
        fm_0 = F.relu(fm_0, inplace=True)
        fm_1 = self.conv_1(neighbor_index, coase_points, fm_0)
        fm_1 = F.relu(fm_1, inplace=True)

        pc = self.upsampling_1([fm_1])

        neighbor_index = get_neighbor_index(pc[1], self.neighbor_num)
        fm_2 = self.conv_2(neighbor_index, pc[1])
        fm_2 = F.relu(fm_2, inplace=True)
        fm_3 = self.conv_3(neighbor_index, pc[1], fm_2)
        fm_3 = F.relu(fm_3, inplace=True)

        fm_4 = self.conv_4(neighbor_index, pc[1], fm_3)
        fm_4 = F.relu(fm_4, inplace=True)
        fm_5 = self.conv_5(neighbor_index, pc[1], fm_4)
        fm_5 = F.relu(fm_5, inplace=True)

        pc_1 = self.upsampling_2([fm_5])




        return pc[1], pc_1[1]

    def getPointcloud(self):
        return self.pointcloud[-1]

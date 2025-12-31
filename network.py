import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # 匈牙利算法

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# 核函数：RBF Gaussian kernel
def gaussian_kernel(x, y, sigma=1.0):
    xx = (x ** 2).sum(dim=1, keepdim=True)
    yy = (y ** 2).sum(dim=1, keepdim=True)
    xy = x @ y.T
    dist = xx + yy.T - 2 * xy
    return torch.exp(-dist / (2 * sigma ** 2))

# 计算 MMD
# def compute_mmd(x, y, sigma=1.0):
#     Kxx = gaussian_kernel(x, x, sigma)
#     Kyy = gaussian_kernel(y, y, sigma)
#     Kxy = gaussian_kernel(x, y, sigma)
#     mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
#     return mmd
def compute_mmd(x, y, sigma=1.0, sample_size=256):
    N = min(x.size(0), y.size(0), sample_size)
    if N < x.size(0):
        idx_x = torch.randperm(x.size(0))[:N]
        idx_y = torch.randperm(y.size(0))[:N]
        x = x[idx_x]
        y = y[idx_y]
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd
# 计算 KL 散度（对称版本）
def compute_kl(x, y, eps=1e-8):
    p = F.softmax(x, dim=1) + eps
    q = F.softmax(y, dim=1) + eps
    kl1 = (p * (p.log() - q.log())).sum(dim=1).mean()
    kl2 = (q * (q.log() - p.log())).sum(dim=1).mean()
    return 0.5 * (kl1 + kl2)

# Main Network
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.device = device

        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim).to(device) for v in range(view)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim).to(device) for v in range(view)])

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

        # α 和 β 作为可学习参数
        self.alpha = nn.Parameter(torch.tensor(1.0, device=device))  # MMD 权重
        self.beta = nn.Parameter(torch.tensor(1.0, device=device))   # KL 权重

    def compute_complementarity_matrix(self, zs):
        V = len(zs)
        C = torch.zeros((V, V), device=self.device)
        for i in range(V):
            for j in range(i + 1, V):
                mmd_val = compute_mmd(zs[i], zs[j])
                kl_val = compute_kl(zs[i], zs[j])
                comp = self.alpha * mmd_val + self.beta * kl_val
                C[i][j] = C[j][i] = comp
        return C

    def optimize_groups(self, C):
        V = C.shape[0]
        original_V = V
        if V % 2 != 0:
            C = F.pad(C, (0, 1, 0, 1), value=0.0)
            V += 1

        cost_matrix = -C.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()
        groups = []

        for i, j in zip(row_ind, col_ind):
            if i >= original_V or j >= original_V or i == j or i in matched or j in matched:
                continue
            groups.append([i, j])
            matched.update([i, j])

        for i in range(original_V):
            if i not in matched:
                groups.append([i])
        return groups

    def hierarchical_fusion(self, zs):
        while len(zs) > 1:
            C = self.compute_complementarity_matrix(zs)
            groups = self.optimize_groups(C)

            new_zs = []
            for group in groups:
                if len(group) == 2:
                    i, j = group
                    fused = (zs[i] + zs[j]) / 2
                else:
                    fused = zs[group[0]]
                new_zs.append(fused)
            zs = new_zs
        return zs[0]

    def feature_fusion(self, zs, zs_gradient):
        H = self.hierarchical_fusion(zs)
        H = self.feature_fusion_module(H)
        return normalize(H, dim=1)

    def compute_fusion_weight(self, z1, z2):
        sim1 = torch.norm(z1, dim=1, keepdim=True)
        sim2 = torch.norm(z2, dim=1, keepdim=True)
        weight1 = sim1 / (sim1 + sim2 + 1e-8)
        weight2 = sim2 / (sim1 + sim2 + 1e-8)
        return weight1, weight2

    def fuse_views(self, zs):
        new_zs = []
        for i in range(self.view):
            for j in range(i + 1, self.view):
                weight1, weight2 = self.compute_fusion_weight(zs[i], zs[j])
                new_z = weight1 * zs[i] + weight2 * zs[j]
                new_zs.append(new_z)
        return new_zs

    def forward(self, xs, zs_gradient=True):
        rs, xrs, zs = [], [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z), dim=1)
            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        new_zs = self.fuse_views(zs)
        new_rs = [normalize(self.common_information_module(new_z), dim=1) for new_z in new_zs]
        H = self.feature_fusion(zs, zs_gradient)

        return xrs, zs, rs, H, new_rs

import os
import torch
import numpy as np
import random
import argparse
from network import Network
from metric import valid
from loss import ContrastiveLoss
from dataloader import load_data
import torch.nn.functional as F
import torch.nn as nn

# 选择数据集
Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置种子与对比训练轮次
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000
    alpha, beta = 0.4, 0.6


# 加载数据
dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)


def compute_view_value(rs, H, view, num_heads=4):
    """
    使用多头注意力机制计算视图权重
    :param rs: 视图特征列表 (list of tensor) -> 每个视图的形状: (N, d_v)
    :param H: 全局特征 (N, d)
    :param view: 视图数 (int)
    :param num_heads: 注意力头数
    :return: 归一化视图权重 (tensor) -> (view,)
    """
    device = H.device
    N, d = H.shape
    d_v = rs[0].shape[1]

    # 多头注意力模块
    mha = nn.MultiheadAttention(embed_dim=d_v, num_heads=num_heads, batch_first=True).to(device)

    Q = H.unsqueeze(1)  # (N, 1, d)
    Q = nn.Linear(d, d_v).to(device)(Q)  # (N, 1, d_v)

    view_scores = []
    for v in range(view):
        K = rs[v].unsqueeze(1)  # (N, 1, d_v)
        attn_output, attn_weights = mha(Q, K, K)  # attn_weights: (N, 1, 1)
        score = attn_weights.mean().item()  # scalar
        view_scores.append(score)

    w = torch.tensor(view_scores, device=device)
    w = F.softmax(w, dim=0)
    return w

# 预训练
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _, _ = model(xs)
        loss_list = [criterion(xs[v], xrs[v]) for v in range(view)]
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')

# 对比训练 (加超参数 alpha, beta)
def contrastive_train(epoch, alpha=1.0, beta=1.0):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H, new_rs = model(xs)
        loss_list = []

        # 融合后对比学习
        with torch.no_grad():
            w = compute_view_value(new_rs, H, len(new_rs))
        for v in range(len(new_rs)):
            loss_list.append(alpha * contrastiveloss(H, new_rs[v], w[v].item()))

        # 未融合对比学习 + 重构
        for v in range(view):
            with torch.no_grad():
                w = compute_view_value(rs, H, view)
            loss_list.append(beta * contrastiveloss(H, rs[v], w[v]))
            loss_list.append(mse(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')
    return tot_loss / len(data_loader)

# 训练与评估
accs, nmis, purs, losses = [], [], [], []
epoch_accuracies, epoch_nmis, epoch_losses = [], [], []

if not os.path.exists('./models'):
    os.makedirs('./models')

# ✅ 去掉循环，直接使用每个数据集设置好的 alpha, beta
print(f"\n==== Running with alpha={alpha:.1f}, beta={beta:.1f} ====")
setup_seed(seed)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

best_acc, best_nmi, best_pur = 0, 0, 0
epoch = 1

while epoch <= args.pre_epochs:
    pretrain(epoch)
    epoch += 1

while epoch <= args.pre_epochs + args.con_epochs:
    loss = contrastive_train(epoch, alpha, beta)
    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
    epoch_accuracies.append(acc)
    epoch_nmis.append(nmi)
    epoch_losses.append(loss)

    if acc > best_acc:
        best_acc, best_nmi, best_pur = acc, nmi, pur
        torch.save(model.state_dict(), f'./models/{args.dataset}_a{alpha:.1f}_b{beta:.1f}.pth')
    epoch += 1

accs.append(best_acc)
nmis.append(best_nmi)
purs.append(best_pur)
print(f'Alpha={alpha:.1f}, Beta={beta:.1f} | Best: ACC={best_acc:.4f}, NMI={best_nmi:.4f}, PUR={best_pur:.4f}')

print("\n================ Final Best Result ================")
print(f'Best Params: alpha={alpha:.1f}, beta={beta:.1f}')
print(f'Best Clustering Performance: ACC={best_acc:.4f}, NMI={best_nmi:.4f}, PUR={best_pur:.4f}')

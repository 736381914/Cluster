import torch, numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 依次计算f中每个元素和所有c之间的欧式距离
# f torch.Size([8, 10, 16, 16])
# c torch.Size([10, 16])
def calc_Eucli_dist_matrix(f, c):
    b, d, h, w = f.shape
    n = h*w
    _, m = c.shape

    f = f.view(b, d, n).permute(0, 2, 1) # 8 256 10
    c = c.permute(1, 0) # 16 10

    f = f.unsqueeze(2).expand(b, n, m, d)
    c = c.unsqueeze(0).expand(n, m, d).unsqueeze(0).expand(b, n, m, d)

    dist_matrix = torch.sqrt(torch.pow(f - c, 2).sum(3))
    return dist_matrix

# 依次计算两两聚类中心之间的欧式距离
# center torch.Size([128, 20])
def calc_center_Eucli_dist_matrix(center):
    epsilon = 1e-2

    d, m = center.shape # 128 20

    center = center.permute(1, 0) # 20 128

    ci = center.unsqueeze(1).expand(m, m, d)
    cj = center.unsqueeze(0).expand(m, m, d)

    # 防止自己和自己的距离为0
    mask = torch.eye(m).unsqueeze(2).expand(m, m, d).cuda() * epsilon
    cj = cj + mask

    center_dist = torch.sqrt(torch.pow(ci - cj, 2).sum(2))
    return center_dist

# f torch.Size([8, 512, 16, 16])
def calc_cov_matrix_np(f):
    b, c, h, w = f.shape  # 8 512 16 16
    f_ = f.view(b, c, -1).cpu().numpy()  # torch.Size([8, 512, 256])
    cov_matrix = torch.zeros(b, c, c).numpy()  # torch.Size([8, 512, 512])
    # 计算协方差矩阵
    for i in range(b):
        cov_matrix[i, :, :] = np.cov(f_[i, :, :])
    cov_matrix = torch.from_numpy(cov_matrix).cuda()  # 对称矩阵
    return cov_matrix

# f torch.Size([8, 512, 16, 16])
def calc_cov_matrix_torch(f):
    b, c, h, w = f.shape  # 8 512 16 16
    f_ = f.view(b, c, -1)  # torch.Size([8, 512, 256])
    cov_matrix = torch.zeros(b, c, c).cuda()  # torch.Size([8, 512, 512])
    # 计算协方差矩阵
    for i in range(b):
        cov_matrix[i, :, :] = torch_cov(f_[i, :, :])
    return cov_matrix

def torch_cov(input_vec):
    x = input_vec - torch.mean(input_vec, dim=1, keepdim=True)
    cov_matrix = torch.matmul(x, x.T) / (x.shape[1]-1)
    return cov_matrix

def TSNE_Visualize(f, labels, center=None):
    f = f.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # center = center.detach().cpu().numpy()
    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)
    # center = tsne.fit_transform(center)
    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5))
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    # ax.scatter(center[:, 0], center[:, 1], c='r')
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

def TSNE_Visualize_diy_color(f, labels):
    f = f.detach().cpu().numpy()
    labels = labels.squeeze(0).detach().cpu().numpy()
    print("f", f.shape)
    print("labels", labels.shape)
    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)
    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5))
    # 设置颜色列表
    colors = ['#c8c925', '#9568bd', '#8c564a', '#d42828', '#1f77b6', '#7e807e', '#2d9f2d', '#e377c2', '#ff8012',
              '#0dbfd0']
    for index in range(10):  # 总共有十个类别，类别的表示为0,1,2...9
        X = data[labels == index][:, 0]
        Y = data[labels == index][:, 1]
        ax.scatter(X, Y, c=colors[index]) # label=index也可以生成好看的颜色
        # ax.scatter(X, Y, label=index) # label=index也可以生成好看的颜色
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

def TSNE_Visualize_center(center):
    center = center.detach().cpu().numpy()
    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, random_state=33)
    center = tsne.fit_transform(center)
    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5))
    ax.scatter(center[:, 0], center[:, 1], c='r')
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

if __name__ == '__main__':
    # f = torch.rand([8,512,16,16])
    # cov_matrix = calc_cov_matrix(f)
    # print(cov_matrix.shape)
    # # 计算协方差通道注意力
    # cov_pool = torch.mean(cov_matrix, dim=2, keepdim=True)
    # cov_weight = torch.sigmoid(cov_pool)
    # print(cov_weight.shape)

    c = torch.rand([2,2]).cuda()
    dist = calc_center_Eucli_dist_matrix(c)
    print(dist, dist.shape)

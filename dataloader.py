from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from scipy.sparse import issparse

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class ThreeRing(Dataset):
    def __init__(self, path):
        # 加载 Three Ring 数据集的 2 个视图数据
        data1 = scipy.io.loadmat(path + 'ThreeRing.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'ThreeRing.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'ThreeRing.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 2 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
def load_data(dataset):
    if dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7


    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

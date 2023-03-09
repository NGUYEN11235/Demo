from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch

class RandomRot:
    def __init__(self, theta=np.pi / 36):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, kp):
        skeleton = kp[:, :, :2]
        T, V, C = skeleton.shape
        results = kp.copy()

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta, self.theta)
            rot_mat = self._rot2d(theta)

        results[:, :, :2] = np.einsum('ab,tvb->tva', rot_mat, skeleton)

        return results



class ActionData(Dataset):
    def __init__(self, anno_path, augmentation=False):
        self.anno = pd.read_pickle(anno_path)
        self.inds_sorted = None
        self.augmentation = augmentation
        self.random_rot = RandomRot()

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        kp = self.anno[index]['pose_results']
        label = self.anno[index]['label']

        if self.augmentation:
            start = np.random.randint(0, len(kp) - 1)
            inds = np.arange(start, start + 1000)
            inds = np.mod(inds, len(kp))
            kp = kp[inds]
            label = label[inds]
            kp = self.random_rot(kp)
        h, w = self.anno[index]['image_size']

        kp[:, :, 0] = kp[:, :, 0] / w
        kp[:, :, 1] = kp[:, :, 1] / h
        ft = torch.from_numpy(kp).float()
        ft = ft.permute(2, 0, 1)

        sample = {'feature': ft, 'label': torch.from_numpy(label).long()}
        return sample
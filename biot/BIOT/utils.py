import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from scipy.signal import resample, butter, iirnotch, filtfilt, lfilter
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

from typing import Literal

class WESADLoader(torch.utils.data.Dataset):
    def __init__(self, files, sensors, n_classes=2, window_size=1, step_size=1, sampling_rate=200, to_seconds=True, 
                 imbalance_dels=0, feat_meth:Literal['resample', 'pca', 'autoencoder']="resample", logpath="log.txt"):
        """
        Parameters:
        ----
        `files`: list of str
        the path to files
        `sensors`:
        the list of sensors
        `n_classes`: 
        number of classes: 2 or 3
        `window_size`: int
        the window size in qsecond, each item size will be (`channels`, `window_size` * `sampling_rate`)
        `step_size`:int
        the step between windows
        `sampling_rate`:int
        the sampling rates would be converted to this
        `to_seconds`:bool
        whether to convert from quarter seconds to seconds
        `feat_meth`: str
        The method to convert features. Used to convert to 200 sampling rate
        """
        self.files = files
        self.sensors = sensors
        self.to_seconds = to_seconds
        self.imbalance_dels = imbalance_dels
        self.n_classes = n_classes
        self.logpath = logpath
        
        self.window = window_size * sampling_rate
        self.sampling_rate = sampling_rate
        self.step_size = step_size * sampling_rate

        if feat_meth == 'resample':
            self.change_freq = self._resample
        elif feat_meth == 'pca':
            for sensor in self.sensors:
                if "chest" not in sensor:
                    raise ValueError("PCA can only be applied on chest signals")
            self.change_freq = self._pca
            
        elif feat_meth == 'autoencoder':
            pass
        else:
            raise AttributeError(f"The `feat_meth` {feat_meth} is not defined")
        
        self.load_files()
        # with open("WESAD_Biot_Xy.pickle", "wb") as file:
        #     pickle.dump((self.Xs, self.Ys), file)
        print(f"Dataset with len of {self.__len__()}")

    def __len__(self):
        return 1 + (self.Ys.shape[1] - self.window) // self.step_size

    def _to_seconds(self, data:torch.Tensor):
        if self.to_seconds:
            return data[:4*(data.shape[0]//4), :].reshape(-1, data.shape[1] * 4)
        else:
            return data
        
    def _resample(self, x):
        return resample(x, self.sampling_rate, axis=1)
    
    def _pca(self, x):
        pca = PCA(self.sampling_rate)
        return pca.fit_transform(x)

    def load_files(self):
        Xs, Ys = [], []
        for file_index in range(len(self.files)):
            with open(self.files[file_index], 'rb') as pklfile:
                # each sample sensor has the shape of (seconds, channels, frequency)
                sample:dict[str, torch.Tensor] = pickle.load(pklfile, encoding='bytes')
            
            # mean(1) converts (#qseconds, channels, freq) -> (#qseconds, freq)
            arrays = [torch.tensor(self.change_freq(sample[sensor].mean(1))).reshape((1, -1)) 
                    for sensor in sample.keys() 
                    if sensor in self.sensors]
            
            X = torch.concat(arrays)
            Y = sample['label'].mode(1).values
            # Y = sample['label'] # in new version label is for each qsecond
            Y = Y.reshape(-1, 1).expand(Y.shape[0], 200).reshape(1, -1) # to 200 sampling rate
            
            X = X / (
                torch.quantile(torch.abs(X), q=0.95, keepdim=True, dim=-1, interpolation="linear")
                # np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            with open(self.logpath, "a") as file:
                file.write(f"Init sizes = X: {X.shape}, Y: {Y.shape}\n")
            
            if self.imbalance_dels:
                Y = Y.squeeze()
                with open(self.logpath, "a") as file:
                    file.write(f"Before imbalance: X: {X.shape}, Y: {Y.shape} (2s: {(Y == 2).sum()}, not2s: {(Y != 2).sum()})\n")
                twos = (Y == 2).nonzero()
                first, last = twos[0].item(), twos[-1].item()
                Y = torch.concat((Y[120000:first-self.imbalance_dels], Y[first:last+1], Y[last+1+self.imbalance_dels:-80000]))
                X = torch.concat((X[:, 120000:first-self.imbalance_dels], X[:, first:last+1], X[:, last+1+self.imbalance_dels:-80000]), dim=1)
                with open(self.logpath, "a") as file:
                    file.write(f"After imbalance: X: {X.shape}, Y: {Y.shape} (2s: {(Y == 2).sum()}, not2s: {(Y != 2).sum()})\n")
                Y = Y.unsqueeze(0)
                
            Xs.append(X)
            Ys.append(Y)
            # with open(self.logpath, "a") as file:
            #     file.write(f"Loading file {file_index}/{len(self.files)}: X = {X.shape}, Y = {Y.shape}\n")
            # self.datasets[file_index] = (X, Y)
        self.Xs = torch.concat(Xs, dim=1)
        self.Ys = torch.concat(Ys, dim=1)
        with open(self.logpath, "a") as file:
            file.write(f"Loaded: Xs: {self.Xs.shape}, Ys: {self.Ys.shape}: (0s: {(self.Ys == 2).sum()}, 1s: {(self.Ys != 2).sum()})\n")
        

        
    def __getitem__(self, index):
        index *= self.step_size
        x = self.Xs[:, index:index + self.window]
        y = self.Ys[:, index:index + self.window]
        y = y.mode().values.item()
        if self.n_classes <= 2: # 0: baseline, 1: stress
            y = 1 if y == 2 else 0
        elif self.n_classes == 3: # 0: baseline, 1: stress, 2: amusement
            y = 1 if y == 2 else 2 if y == 3 else 0
        y = torch.tensor([y])
        with open(self.logpath, "a") as file:
            file.write(f"___get item = x: {x.shape}, y: {y.shape}, i = {index}, w = {self.window}\n")
        
        return x.type(torch.float32), y.type(torch.float32)

def collate_fn_WESAD_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples



class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()

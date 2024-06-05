import os
import argparse
import pickle

import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from .model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from .utils import WESADLoader, TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.train_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y)  # focal_loss(prob, y)
        self.log("train_loss", loss)
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.val_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_WESAD_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = os.path.abspath(os.curdir) + "/WESAD/"
    file_path = root + "S{s}/S{s}_n0.pkl"
    train_files = [file_path.format(s=i)  for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = [file_path.format(s=i) for i in [13, 14, 15, 16]]
    test_files = [file_path.format(s=i) for i in [17]]

    print(f"Nom of:\n\tTrain: {len(train_files)}\n\tVal:   {len(val_files)}\n\tTest:  {len(test_files)}")

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        WESADLoader(train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        WESADLoader(test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        WESADLoader(val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(f"Size of loader of:\n\tTrain: {len(train_loader)}\n\tVal:   {len(val_loader)}\n\tTest:  {len(test_loader)}")
    return train_loader, test_loader, val_loader

def prepare_WESAD_dataloader_manual(sampling_rate=200, batch_size=512, num_workers=4):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = os.path.abspath(os.curdir) + "/WESAD/"
    file_path = root + "S{s}/S{s}_n0.pkl"
    train_files = [file_path.format(s=i)  for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = [file_path.format(s=i) for i in [13, 14, 15, 16]]
    test_files = [file_path.format(s=i) for i in [17]]

    print(f"Len of train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        WESADLoader(train_files, sampling_rate),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        WESADLoader(test_files, sampling_rate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        WESADLoader(val_files, sampling_rate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(f"train size: {len(train_loader)}, val size: {len(val_loader)}, test size: {len(test_loader)}")
    return train_loader, test_loader, val_loader


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/WFDB/processed2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def supervised(args):
    print("Welcome")
    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    elif args.dataset == "WESAD":
        print(f"Dataset: WESAD")
        train_loader, test_loader, val_loader = prepare_WESAD_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="cpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers")
    parser.add_argument("--dataset", type=str, default="WESAD", help="dataset")
    parser.add_argument(
        "--model", type=str, default="BIOT", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    supervised(args)

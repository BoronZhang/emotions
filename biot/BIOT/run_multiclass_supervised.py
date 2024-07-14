import os
import argparse
import pickle
import random

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import multiclass_metrics_fn

try:
    from model import (
        SPaRCNet,
        ContraWR,
        CNNTransformer,
        FFCL,
        STTransformer,
        BIOTClassifier,
    )
    from utils import TUEVLoader, HARLoader, WESADLoader
except:
    from .model import (
        SPaRCNet,
        ContraWR,
        CNNTransformer,
        FFCL,
        STTransformer,
        BIOTClassifier,
    )
    from .utils import TUEVLoader, HARLoader, WESADLoader


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        

    def training_step(self, batch, batch_idx):
        X, y = batch
        prod = self.model(X)
        loss = nn.CrossEntropyLoss()(prod, y.squeeze().type(torch.LongTensor).to(torch.device(self.args.device)))
        self.log("train_loss", loss)
        self.train_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.train_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.train_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()
        
        self.val_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = []
        gt = np.array([])
        for out in self.val_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val_f1", result["f1_weighted"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = []
        gt = np.array([])
        for out in self.test_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)

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
    random.seed(seed)

    root = os.path.abspath(os.curdir) + "/WESAD/"
    if args.server == "pc":
        file_path = root + "S{s}/S{s}_n0.pkl"
    elif args.server == "colab":
        file_path = root + "S{s}_n0.pkl"
    elif args.server == "kaggle":
        file_path = "S{s}_n0.pkl"
    All_files = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    All_files = [i for i in All_files if i != args.test]
    val_indices = random.sample(range(len(All_files)), k=4)
    train_files = [file_path.format(s=All_files[i])  for i in range(len(All_files)) if i not in val_indices]
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = [file_path.format(s=All_files[i])  for i in range(len(All_files)) if i in val_indices]
    test_files = [file_path.format(s=i) for i in [args.test]]
    # print(f"Train files: {train_files}\nValid: {val_files}\nTest files: {test_files}")

    # print(f"Nom of:\n\tTrain: {len(train_files)}\n\tVal:   {len(val_files)}\n\tTest:  {len(test_files)}")

    # prepare training and test data loader
    loader_args = {'sampling_rate': args.sampling_rate, 'window_size': args.window_size, 'step_size': args.step_size, 
                   "sensors": args.sensors, "imbalance_dels": args.imbalance_dels, 'logpath': args.logpath, 'feat_meth': args.feat_meth}
    print(f"Experiment with sensors:\n")
    for s in args.sensors:
        print(f"\t- {s}")
    train_loader = torch.utils.data.DataLoader(
        WESADLoader(files=train_files, **loader_args),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        WESADLoader(files=test_files, **loader_args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        WESADLoader(files=val_files, **loader_args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    # print(f"Size of loader of:\n\tTrain: {len(train_loader)}\n\tVal:   {len(val_loader)}\n\tTest:  {len(test_loader)}")
    return train_loader, test_loader, val_loader


def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_HAR_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/HAR/processed/"

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


class Supervised:
    def __init__(self, args) -> None:
        self.args = args
    def supervised_go(self):
        args = self.args
        # get data loaders
        if args.dataset == "TUEV":
            train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)
        elif args.dataset == "WESAD":
            train_loader, test_loader, val_loader = prepare_WESAD_dataloader(args)

        else:
            raise NotImplementedError

        # define the model
        if args.model == "SPaRCNet":
            model = SPaRCNet(
                in_channels=args.in_channels,
                sample_length=int(args.sample_length * args.sampling_rate),
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
                n_segments=4 if args.dataset == "HAR" else 5,
            )

        elif args.model == "FFCL":
            model = FFCL(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.token_size,
                steps=args.hop_length // 5,
                sample_length=int(args.sample_length * args.sampling_rate),
                shrink_steps=16 if args.dataset == "HAR" else 20,
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
            save_dir= "../../working/" if args.server == "kaggle" else "./",
            version=version,
            name="log",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_cohen", patience=5, verbose=False, mode="max"
        )

        trainer = pl.Trainer(
            devices="auto",
            accelerator=args.device,
            strategy=DDPStrategy(find_unused_parameters=False),
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
            model=lightning_model, dataloaders=test_loader
        )[0]
        print(pretrain_result)
        return pretrain_result


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
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=12, help="number of input channels"
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
    print(args)

    supervised(args)

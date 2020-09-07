import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, RandomSampler

from preprocessor import Preprocessor
from dataset import NerDataset
from net import NerBertModel


def get_dataloader(data_path, preprocessor, batch_size):
    train_dataset = NerDataset(os.path.join(data_path, "train_data.txt"), preprocessor)
    val_dataset = NerDataset(os.path.join(data_path, "val_data.txt"), preprocessor)
    test_dataset = NerDataset(os.path.join(data_path, "test_data.txt"), preprocessor)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def main(args):
    preprocessor = Preprocessor(args.bert_model, args.max_len)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        args.data_path, preprocessor, args.batch_size
    )

    bert_finetuner = NerBertModel(
        args, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=args.log_path, version=1, name=args.task)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/" + args.task + "/{epoch}_{val_acc:3f}",
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        prefix="",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode="max",
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        # distributed_backend="",
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
    )

    trainer.fit(bert_finetuner)
    trainer.test()


if __name__ == "__main__":
    config = OmegaConf.load("conf/train_config.yaml")
    main(config)
import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, RandomSampler
from itertools import chain

from preprocessor import Preprocessor
from dataset import JointDataset
from net import NerBertModel


def get_slot_labels(data_path):
    tag_list = [
        line.rstrip("\n").split(" ")
        for line in open(
            os.path.join(data_path, "train", "seq.out"), mode="r", encoding="utf-8"
        )
    ]

    slot_labels = list(set(list(chain.from_iterable(tag_list))))
    slot_labels = sorted(slot_labels, key=lambda x: (x[2:], x[:2]))
    slot_labels = ["UNK", "PAD"] + slot_labels

    return slot_labels


def get_intent_labels(data_path):
    intent_list = [
        line.rstrip("\n")
        for line in open(
            os.path.join(data_path, "train", "label"), mode="r", encoding="utf-8"
        )
    ]

    intent_labels = list(set(intent_list))
    intent_labels = sorted(intent_labels)
    intent_labels = ["UNK"] + intent_labels

    return intent_labels


def get_dataloader(config, preprocessor):
    train_dataset = JointDataset(config, preprocessor, "train")
    val_dataset = JointDataset(config, preprocessor, "dev")
    test_dataset = JointDataset(config, preprocessor, "test")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, drop_last=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def main(config):
    config.slot_labels = get_slot_labels(config.data_path)
    config.intent_labels = get_intent_labels(config.data_path)

    preprocessor = Preprocessor(config.bert_model, config.max_len)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        config, preprocessor
    )

    bert_finetuner = NerBertModel(
        config, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=config.log_path, version=1, name=config.task)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/" + config.task + "/{epoch}_{val_acc:3f}",
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        prefix="",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=config.gpus,
        # distributed_backend="",
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
    )

    trainer.fit(bert_finetuner)
    trainer.test()


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)
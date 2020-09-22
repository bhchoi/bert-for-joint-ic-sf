import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, RandomSampler

from preprocessor import Preprocessor
from dataset import JointDataset
from net import NerBertModel


def get_dataloader(config, preprocessor):
    train_dataset = JointDataset(config, preprocessor, "train")
    val_dataset = JointDataset(config, preprocessor, "dev")
    test_dataset = JointDataset(config, preprocessor, "test")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, sampler=train_sampler
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def main(config):

    preprocessor = Preprocessor(config.bert_model, config.max_len)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        config, preprocessor
    )

    bert_finetuner = NerBertModel(
        config, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=config.log_path, name=config.task)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/" + config.task + "/{epoch}_{val_loss:.3f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
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
        distributed_backend=config.distributed_backend,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
    )

    trainer.fit(bert_finetuner)
    trainer.test()


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)
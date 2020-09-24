import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import Storage
from torch.utils.data import DataLoader, RandomSampler

from preprocessor import Preprocessor
from dataset import JointDataset
from net import NerBertModel


def get_dataloader(config, preprocessor):
    test_dataset = JointDataset(config, preprocessor, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size)

    return test_dataloader


def main(config):

    preprocessor = Preprocessor(config.bert_model, config.max_len)
    test_dataloader = get_dataloader(config, preprocessor)
    model = NerBertModel(config, None, None, test_dataloader)
    checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    
    trainer = pl.Trainer()
    res = trainer.test(model)


if __name__ == "__main__":
    config = OmegaConf.load("config/eval_config.yaml")
    main(config)
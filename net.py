import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import os
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score as seqeval_f1_score
from sklearn.metrics import f1_score as sklearn_f1_score
from itertools import chain

from utils import get_intent_labels
from utils import get_slot_labels


class NerBertModel(pl.LightningModule):
    def __init__(
        self,
        config: argparse,
        ner_train_dataloader: DataLoader,
        ner_val_dataloader: DataLoader,
        ner_test_dataloader: DataLoader,
    ):
        super().__init__()
        self.config = config
        self.ner_train_dataloader = ner_train_dataloader
        self.ner_val_dataloader = ner_val_dataloader
        self.ner_test_dataloader = ner_test_dataloader
        self.ignore_index = torch.nn.CrossEntropyLoss().ignore_index
        self.total_intent_labels = get_intent_labels(
            os.path.join(self.config.data_path, self.config.task)
        )
        self.total_slot_labels = get_slot_labels(
            os.path.join(self.config.data_path, self.config.task)
        )

        self.bert_config = BertConfig.from_pretrained(self.config.bert_model)
        self.model = BertModel.from_pretrained(self.config.bert_model)
        self.intent_dropout = nn.Dropout(self.config.dropout_rate)
        self.intent_linear = nn.Linear(
            self.bert_config.hidden_size, len(self.total_intent_labels)
        )
        self.slot_dropout = nn.Dropout(self.config.dropout_rate)
        self.slot_linear = nn.Linear(
            self.bert_config.hidden_size, len(self.total_slot_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        slot_output = outputs[0]
        slot_output = self.slot_dropout(slot_output)
        slot_output = self.slot_linear(slot_output)

        intent_output = outputs[1]
        intent_output = self.intent_dropout(intent_output)
        intent_output = self.intent_linear(intent_output)

        return slot_output, intent_output

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels, intent_labels = batch

        slot_output, intent_output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        active_loss = attention_mask.view(-1) == 1
        active_logits = slot_output.view(-1, len(self.total_slot_labels))[active_loss]
        active_labels = slot_labels.view(-1)[active_loss]
        slot_loss = F.cross_entropy(active_logits, active_labels)

        intent_loss = F.cross_entropy(
            intent_output.view(-1, len(self.total_intent_labels)),
            intent_labels.view(-1),
        )

        tensorboard_logs = {
            "train_loss": slot_loss + intent_loss,
            "train_intent_loss": intent_loss,
            "train_slot_loss": slot_loss,
        }

        train_outputs = {"loss": slot_loss + intent_loss, "log": tensorboard_logs}

        return train_outputs

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels, intent_labels = batch

        slot_output, intent_output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        active_loss = attention_mask.view(-1) == 1
        active_logits = slot_output.view(-1, len(self.total_slot_labels))[active_loss]
        active_labels = slot_labels.view(-1)[active_loss]
        slot_loss = F.cross_entropy(active_logits, active_labels)

        a, y_hat = torch.max(slot_output, dim=2)
        y_hat = y_hat.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.total_slot_labels)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.ignore_index:
                    slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        val_slot_acc = torch.tensor(
            seqeval_f1_score(slot_gt_labels, slot_pred_labels), dtype=torch.float32
        )
        token_val_slot_acc = sklearn_f1_score(
            list(chain.from_iterable(slot_gt_labels)),
            list(chain.from_iterable(slot_pred_labels)),
            average="micro",
        )

        token_val_slot_acc = torch.tensor(token_val_slot_acc, dtype=torch.float32)

        intent_loss = F.cross_entropy(
            intent_output.view(-1, len(self.total_intent_labels)),
            intent_labels.view(-1),
        )

        _, pred_intent_labels = torch.max(intent_output, dim=1)

        intent_labels = intent_labels.detach().cpu().numpy()
        pred_intent_labels = pred_intent_labels.detach().cpu().numpy()

        # TODO: cpu
        intent_acc = (intent_labels == pred_intent_labels).mean()
        intent_acc = torch.tensor(intent_acc)

        validation_outputs = {
            "val_slot_loss": slot_loss,
            "val_intent_loss": intent_loss,
            "val_loss": slot_loss,
            "val_slot_acc": val_slot_acc,
            "token_val_slot_acc": token_val_slot_acc,
            "val_intent_acc": intent_acc,
        }

        return validation_outputs

    def validation_epoch_end(self, outputs):
        val_slot_loss = torch.stack([x["val_slot_loss"] for x in outputs]).mean()
        val_intent_loss = torch.stack([x["val_intent_loss"] for x in outputs]).mean()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_slot_acc = torch.stack([x["val_slot_acc"] for x in outputs]).mean()
        token_val_slot_acc = torch.stack(
            [x["token_val_slot_acc"] for x in outputs]
        ).mean()
        val_intent_acc = torch.stack([x["val_intent_acc"] for x in outputs]).mean()

        tensorboard_log = {
            "val_loss": val_loss,
            "val_slot_loss": val_slot_loss,
            "val_intent_loss": val_intent_loss,
            "val_slot_acc": val_slot_acc,
            "token_val_slot_acc": token_val_slot_acc,
            "val_intent_acc": val_intent_acc,
        }

        return {"val_loss": val_loss, "progress_bar": tensorboard_log}

    def test_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels, intent_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, len(self.total_slot_labels))[active_loss]
        active_labels = outputs.view(-1)[active_loss]
        loss = F.cross_entropy(active_logits, active_labels)

        a, y_hat = torch.max(outputs, dim=2)
        y_hat = y_hat.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.total_slot_labels)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.ignore_index:
                    slot_gt_labels[i].append(slot_label_map[slot_gt_labels[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        test_acc = torch.tenfor(
            seqeval_f1_score(slot_gt_labels, slot_pred_labels), dtype=torch.float32
        )
        token_test_acc = sklearn_f1_score(
            list(chain.from_iterable(slot_gt_labels)),
            list(chain.from_iterable(slot_pred_labels)),
            average="micro",
        )

        token_test_acc = torch.tensor(token_test_acc, dtype=torch.float32)

        test_step_outputs = {
            "test_acc": test_acc,
            "token_test_acc": token_test_acc,
            "gt_labels": slot_gt_labels,
            "pred_labels": slot_pred_labels,
        }

        return test_step_outputs

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        token_test_acc = torch.stack([x["token_test_acc"] for x in outputs]).mean()

        gt_labels = []
        pred_labels = []
        for x in outputs:
            gt_labels.extend(x["gt_labels"])
            pred_labels.extend(x["pred_labels"])

        test_step_outputs = {
            "test_acc": test_acc,
            "token_test_acc": token_test_acc,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
        }

        return test_step_outputs

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.ner_train_dataloader

    def val_dataloader(self):
        return self.ner_val_dataloader

    def test_dataloader(self):
        return self.ner_test_dataloader
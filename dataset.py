import os
from torch.utils.data import Dataset

from preprocessor import Preprocessor
from utils import get_intent_labels
from utils import get_slot_labels


class JointDataset(Dataset):
    def __init__(self, config, preprocessor, mode):
        self.config = config
        self.mode = mode
        self.sentence_list = []
        self.tag_list = []
        self.intent_list = []
        self.preprocessor = preprocessor
        self.intent_labels = get_intent_labels(self.config.data_path)
        self.slot_labels = get_slot_labels(self.config.data_path)

        self.load_data()

    def load_data(self):

        self.sentence_list = [
            line.rstrip("\n").split()
            for line in open(
                os.path.join(self.config.data_path, self.mode, "seq.in"),
                mode="r",
                encoding="utf-8",
            )
        ]
        self.tag_list = [
            line.rstrip("\n").split()
            for line in open(
                os.path.join(self.config.data_path, self.mode, "seq.out"),
                mode="r",
                encoding="utf-8",
            )
        ]
        self.intent_list = [
            line.rstrip("\n")
            for line in open(
                os.path.join(self.config.data_path, self.mode, "label"),
                mode="r",
                encoding="utf-8",
            )
        ]

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        sentence = self.sentence_list[idx]

        intent = self.intent_list[idx]

        intent_id = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")

        tags = [
            self.slot_labels.index(t)
            if t in self.slot_labels
            else self.slot_labels.index("UNK")
            for t in self.tag_list[idx]
        ]

        return self.preprocessor.get_input_features(sentence, tags, intent_id)

import os
from preprocessor import Preprocessor
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, config, preprocessor):
        self.config = config
        self.sentence_list = []
        self.tag_list = []
        self.intent_list = []
        self.preprocessor = preprocessor

        self.load_data()

    def load_data(self):

        self.sentence_list = [
            line.rstrip("\n").split(" ")
            for line in open(
                os.path.join(self.config.data_path, "seq.in"),
                mode="r",
                encoding="utf-8",
            )
        ]
        self.tag_list = [
            line.rstrip("\n").split(" ")
            for line in open(
                os.path.join(self.config.data_path, "seq.out"),
                mode="r",
                encoding="utf-8",
            )
        ]
        self.intent_list = [
            line.rstrip("\n")
            for line in open(
                os.path.join(self.config.data_path, "label"), mode="r", encoding="utf-8"
            )
        ]

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        sentence = self.sentence_list[idx]
        intent = self.intent_list[idx]
        tags = [
            self.config.slot_labels.index(t)
            if t in self.config.slot_labels
            else self.preprocessor.tokenizer.unk_token
            for t in self.tag_list[idx]
        ]

        # TODO: get_input_features에 intent 추가하기
        (
            input_ids,
            slot_labels,
            attention_mask,
            token_type_ids,
        ) = self.preprocessor.get_input_features(sentence, tags)

        return input_ids, slot_labels, attention_mask, token_type_ids

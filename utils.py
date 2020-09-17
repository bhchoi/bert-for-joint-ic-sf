import os
from itertools import chain


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
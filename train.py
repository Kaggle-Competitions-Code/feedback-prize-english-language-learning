# -*- encoding: utf-8 -*-
'''
@create_time: 2022/10/17 13:41:04
@author: lichunyu
'''
import os
import sys
import logging
from logging.config import dictConfig
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    default_data_collator
)
from tqdm import tqdm


_logconfig_dict_default = {
    "version": 1,
    "incremental": False,
    "disable_existing_loggers": False,
    "root": {
        "level": "INFO",
        "handlers": ["timed_rotating_file_handler", "console"]
    },
    "formatters": {
        "default": {
            "class": "logging.Formatter",
            "format": "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S %z"
        }
    },
    "handlers": {
        "timed_rotating_file_handler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": "log.log",
            "when": "D",
            "backupCount": 7,
            "interval": 1,
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {}
}

dictConfig(_logconfig_dict_default)

logger = logging.getLogger()


_default_tokenizer_param = {
    "add_special_tokens": True,
    "truncation": "longest_first",
    "max_length": 400,
    "padding": "max_length",
    "return_attention_mask": True,
    "return_tensors": "pt"
}


class TrainDataset(Dataset):

    def __init__(self, data_path, tokenizer) -> None:
        data = pd.read_csv(data_path)
        self.cohesion = data["cohesion"].values
        self.full_text = data["full_text"].values
        self.syntax = data["syntax"].values
        self.phraseology = data["phraseology"].values
        self.grammar = data["grammar"].values
        self.conventions = data["conventions"].values
        self.tokenizer = tokenizer
        self._tokenizer_all()

    def _tokenizer_all(self):
        result = []
        for idx, text in enumerate(self.full_text):
            ...

    def __len__(self):
        return len(self.full_text)

    def __getitem__(self, index):
        return super().__getitem__(index)


class Trainer(object):

    def __init__(self, model, config, train_datasets, dev_datasets) -> None:
        self.model = model
        self.train_datasets = train_datasets
        self.dev_datasets = dev_datasets
        self.config = self.config

    def _setup(self):
        if self.config.mode == None:
            self.train_dataloader = DataLoader(self.train_datasets, batch_sampler=)


    def train(self):
        for epoch in range(self.config.epoch):
            for batch in tqdm()





def main(json_path):
    parser = HfArgumentParser()
    if json_path:
        advance_args, = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        advance_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        advance_args, = parser.parse_args_into_dataclasses()



if __name__ == "__main__":
    ...
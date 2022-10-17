# -*- encoding: utf-8 -*-
'''
@create_time: 2022/10/17 13:17:32
@author: lichunyu
'''
from transformers import DebertaConfig, DebertaModel, AutoModel, AutoTokenizer, DebertaV2Tokenizer, BertModel
import pandas as pd

model = AutoModel.from_pretrained("/root/pretrained_model/deberta-v3-large")
# tokenizer = DebertaV2Tokenizer.from_pretrained("/root/pretrained_model/deberta-v3-large", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)


_default_tokenizer_param = {
    "add_special_tokens": True,
    "truncation": "longest_first",
    "max_length": 1002,
    "padding": "max_length",
    "return_attention_mask": True,
    "return_tensors": "pt"
}

# df = pd.read_csv("/root/feedback-prize-english-language-learning/data/train.csv")
batch = tokenizer(
    "Every 2s refresh data",
    **_default_tokenizer_param
)
output = model(**batch)
...
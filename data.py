import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule, Trainer 
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import (
    DistilBertTokenizer,
    DistilBertPreTrainedModel,
    DistilBertModel,
    AdamW,
    DistilBertConfig
)
from scipy.special import softmax, logit 
from sklearn.linear_model import LogisticRegression
from sklearn import cross_decomposition
import math 
import numpy as np 
import tqdm 
import argparse
import pandas as pd
from collections import defaultdict
import warnings 
warnings.filterwarnings("ignore")

class DataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=2, debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.df = pd.read_csv(csv_path)
        if debug: self.df = self.df.sample(20)
        self.texts = self.df['text']
        self.confounds = self.df['C']
        if 'T' in self.df.columns: self.treatments = self.df['T']
        else: self.treatments = None
        if 'Y' in self.df.columns : self.outcomes = self.df['Y']
        else: self.outcomes = None

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased', do_lower_case=True
        )

        if self.treatments is None:
            self.treatments = [-1 for _ in range(len(self.confounds))]
        if self.outcomes is None:
            self.outcomes = [-1 for _ in range(len(self.treatments))]

        self.out = defaultdict(list)


    def setup(self, stage):
        if stage=='fit':
            for i, (W, C, T, Y) in enumerate(zip(self.texts, self.confounds, self.treatments, self.outcomes)):
                encoded_sent = self.tokenizer.encode_plus(
                    W, add_special_tokens=True,
                    max_length=128,
                    truncation=True,
                    pad_to_max_length=True
                )

                self.out['W_ids'].append(encoded_sent['input_ids'])
                self.out['W_mask'].append(encoded_sent['attention_mask'])
                self.out['W_len'].append(sum(encoded_sent['attention_mask']))
                self.out['Y'].append(Y)
                self.out['T'].append(T)
                self.out['C'].append(C)
                # self.out['use_mlm'].append(True)
            data = (torch.tensor(self.out[x]) for x in ['W_ids', 'W_len', 'W_mask', 'C', 'T', 'Y'])
            self.train_data = TensorDataset(*data)
            self.train_sampler = RandomSampler(self.train_data)

        if stage=='predict':
            for i, (W, C, T, Y) in enumerate(zip(self.texts, self.confounds, self.treatments, self.outcomes)):
                encoded_sent = self.tokenizer.encode_plus(
                    W, add_special_tokens=True,
                    max_length=128,
                    truncation=True,
                    pad_to_max_length=True
                )

                self.out['W_ids'].append(encoded_sent['input_ids'])
                self.out['W_mask'].append(encoded_sent['attention_mask'])
                self.out['W_len'].append(sum(encoded_sent['attention_mask']))
                # self.out['Y'].append(Y)
                self.out['T'].append(T)
                self.out['C'].append(C)
            data = (torch.tensor(self.out[x]) for x in ['W_ids', 'W_len', 'W_mask', 'C', 'T'])
            self.predict_data = TensorDataset(*data)
            self.predict_sampler = SequentialSampler(self.predict_data)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, sampler=self.train_sampler, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, sampler=self.test_sampler, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, sampler=self.predict_sampler, batch_size=self.batch_size)


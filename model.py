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
    DistilBertConfig,
    get_linear_schedule_with_warmup
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
from rich.console import Console 
from rich.table import Table
import warnings 
warnings.filterwarnings("ignore")

class CausalBERT(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        g_weight=0.1,
        q_weight=0.1,
        mlm_weight=1.0,
    ):
        super().__init__()

        self.config = DistilBertConfig.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(
            model_name,
            output_attentions=False,
            output_hidden_states=False
        )
        self.num_labels = num_labels
        self.vocab_size = self.config.vocab_size
        self.g_weight = g_weight
        self.q_weight = q_weight
        self.mlm_weight = mlm_weight

        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)

        self.Q_cls = nn.ModuleDict()

        self.MASK_IDX = 103

        for i in range(2):
            self.Q_cls[f"{i}"] = nn.Sequential(
                nn.Linear(self.config.hidden_size+self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, self.num_labels)
            )
        
        self.g_cls = nn.Linear(self.config.hidden_size+self.num_labels, self.config.num_labels)

    # def test_step(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):
    def training_step(self, batch, batch_idx):
        W_ids, W_len, W_mask, C, T, Y = batch
        # if use_mlm:
        W_len = W_len.unsqueeze(1) - 2
        mask_class = torch.empty(W_len.shape, dtype=torch.float).uniform_()
        mask_class = mask_class.to(W_len)
        mask = (mask_class * W_len.float()).long() + 1
        target_words = torch.gather(W_ids, 1, mask)
        mlm_labels = torch.ones(W_ids.shape).long() * -100
        mlm_labels.to(W_ids)
        mlm_labels.scatter_(1, mask, target_words)
        W_ids.scatter_(1, mask, self.MASK_IDX)
        
        outputs = self.bert(W_ids, attention_mask=W_mask)
        seq_output = outputs[0]
        pooled_output = seq_output[:, 0]

        # if use_mlm:
        pred_logits = self.vocab_transform(seq_output)
        pred_logits = F.gelu(pred_logits)
        pred_logits = self.vocab_layer_norm(pred_logits)
        pred_logits = self.vocab_projector(pred_logits)
        mlm_loss = nn.CrossEntropyLoss()(
            pred_logits.view(-1, self.vocab_size), mlm_labels.view(-1)
        )
        # else:
        #     mlm_loss = 0.0

        C_bow = self._make_bow_vector(C.unsqueeze(1), self.num_labels)
        inputs = torch.cat((pooled_output, C_bow), 1)

        g = self.g_cls(inputs)


        g_loss = nn.CrossEntropyLoss()(g.view(-1, self.num_labels), T.view(-1))        
        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)


        T0_indices = (T==0).nonzero().squeeze()
        Y_T1_labels = Y.clone().scatter(0, T0_indices, -100)
        T1_indices = (T==1).nonzero().squeeze()
        Y_T0_labels = Y.clone().scatter(0, T1_indices, -100)

        Q_loss_T1 = nn.CrossEntropyLoss()(
            Q_logits_T1.view(-1, self.num_labels), Y_T1_labels
        )
        Q_loss_T0 = nn.CrossEntropyLoss()(
            Q_logits_T0.view(-1, self.num_labels), Y_T0_labels
        )


        Q_loss = Q_loss_T0 + Q_loss_T1
        
        sm = nn.Softmax(dim=1)
        Q0 = sm(Q_logits_T0)[:, 1]
        Q1 = sm(Q_logits_T1)[:, 1]
        g = sm(g)[:, 1]


        # return g, Q0, Q1, g_loss, Q_loss, mlm_loss
        loss = self.g_weight * g_loss + \
                self.q_weight * Q_loss + self.mlm_weight * mlm_loss
        return loss


    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,

        # )
        return AdamW(self.parameters(), lr=2e-5, eps=1e-8)

    def forward(self, inputs):
        W_ids, W_len, W_mask, C, T = inputs
        outputs = self.bert(W_ids, attention_mask=W_mask)
        seq_output = outputs[0]
        pooled_output = seq_output[:, 0]
        C_bow = self._make_bow_vector(C.unsqueeze(1), self.num_labels)
        inputs = torch.cat((pooled_output, C_bow), 1)

        # g logits
        g = self.g_cls(inputs)

        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        sm = nn.Softmax(dim=1)
        Q0 = sm(Q_logits_T0)[:, 1]
        Q1 = sm(Q_logits_T1)[:, 1]
        g = sm(g)[:, 1]

        # print(f"{g=}")
        # print(f"{Q0=}")
        # print(f"{Q1=}")

        # probs = np.array(list(zip(Q0, Q1)))
        # Q0 = probs[:, 0]
        # Q1 = probs[:, 1]

        result = torch.vstack((g, Q0, Q1))
        result = result.transpose(1, 0)
        return result.detach().cpu().numpy()
    # def predict_step(self, batch, batch_idx):
    #     W_ids, W_len, W_mask, C, T = batch 
    #     outputs = self.bert(W_ids, attention_mask=W_mask)
    #     seq_output = outputs[0]
    #     pooled_output = seq_output[:, 0]
    #     C_bow = self._make_bow_vector(C.unsqueeze(1), self.num_labels)
    #     inputs = torch.cat((pooled_output, C_bow), 1)
    #     g = self.g_cls(inputs)
    #     Q_logits_T0 = self.Q_cls['0'](inputs)
    #     Q_logits_T1 = self.Q_cls['1'](inputs)

    #     sm = nn.Softmax(dim=1)
    #     Q0 = sm(Q_logits_T0)[:, 1]
    #     Q1 = sm(Q_logits_T1)[:, 1]
    #     g = sm(g)[:, 1]

    #     return g.detach().cpu().item(), Q0.detach().cpu().item(), Q1.detach().cpu().item() 

    def on_predict_epoch_end(self, results):
        results = np.array(results)
        results = results.reshape(-1, 3) # shape=(steps, batchsize, 3) -> (steps*batchsize, 3)
        # results = results[:,:,1] # shape=(b, 3[g,q0,q1], 2) -> (b, 3[g, q0, q1], 1)
        # print(f"{results.shape}")
        # results = results.squeeze(0)
        # print(f"{results=}")
        g = results[:, 0]
        Q0 = results[:, 1]
        Q1 = results[:, 2]
        # return np.mean(g), np.mean(Q0), np.mean(Q1), np.mean(Q0-Q1)

        console = Console()
        table = Table(
            show_header=True,
            header_style='bold magenta',
        )
        table.add_column("Estimate", justify='center')
        table.add_column("Predicted value", justify='center')
        # table.add_column("Reproduced value")

        results_dict = {
            '[blue]g': str(np.round(np.mean(g), 2)),
            '[blue]Q0': str(np.round(np.mean(Q0), 2)),
            '[blue]Q1': str(np.round(np.mean(Q1), 2)),
            '[green]NDE': str(np.round(np.mean(Q0 - Q1), 2))
        }


        for key, val in results_dict.items():
            table.add_row(
                key,
                val
            )
        console.print(table)
        # return np.mean(Q0 - Q1)
    
    def _make_bow_vector(self, ids, vocab_size, use_counts=False):
        vec = torch.zeros(ids.shape[0], vocab_size)
        ones = torch.ones_like(ids, dtype=torch.float)
        vec.scatter_add_(1, ids, ones)
        vec[:, 1] = 0.0
        if not use_counts:
            vec = (vec != 0).float()
        return vec
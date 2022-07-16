import os
import gc
import re
import sys
sys.path.append("/root/workspace/FeedbackPrize2")
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import librosa
import scipy as sp
import numpy as np
import pandas as pd
import soundfile as sf
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
tqdm.pandas()

from contextlib import contextmanager
from pathlib import Path
from typing import List
from typing import Optional
from sklearn import metrics
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import transformers
import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle


INPUT_DIR = 'input/'


def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def get_score(outputs, labels):
    outputs = F.softmax(torch.tensor(outputs)).numpy()
    return log_loss(labels, outputs)


class CFG:
    EXP_ID = '013'
    apex = True
    model = 'microsoft/deberta-v3-large'  # 'microsoft/deberta-v3-base'
    seed = 71
    n_splits = 4
    max_len = 512
    dropout = 0 # 0.2
    target_size=3
    n_accumulate=1
    print_freq = 100
    eval_freq = 1700 # 1200
    min_lr=1e-6
    scheduler = 'cosine'
    batch_size = 8 # 16 # 8
    num_workers = 3
    lr = 3e-6
    weigth_decay = 0.01
    epochs = 4 # 5 # 4
    n_fold = 4
    trn_fold = [i for i in range(n_fold)]
    train = True 
    num_warmup_steps = 0
    num_cycles=0.5
    debug = False # True
    freezing = True
    gradient_checkpoint = True
    # itpt_path = 'itpt/deberta_v3_large'
    reinit_layers = 4 # 3
    # max_norm = 0.5
    text_features = [
            'discourse_text_num_chars', 'discourse_text_num_capitals', 'discourse_text_caps_vs_length', 
            'discourse_text_num_exclamation_marks', 'discourse_text_num_question_marks', 
            'discourse_text_num_punctuation', 'discourse_text_num_symbols', 
            'discourse_text_num_words', 'discourse_text_num_unique_words', 'discourse_text_words_vs_unique', 
            'essay_text_num_chars', 'essay_text_num_capitals', 'essay_text_caps_vs_length', 'essay_text_num_exclamation_marks', 
            'essay_text_num_question_marks', 'essay_text_num_punctuation', 'essay_text_num_symbols', 'essay_text_num_words', 
            'essay_text_num_unique_words', 'essay_text_words_vs_unique',
            'div_discourse_text_num_chars_essay_text_num_chars', 'div_discourse_text_num_words_essay_text_num_words', 'div_discourse_text_num_unique_words_essay_text_num_unique_words'
    ] # 23 features


if CFG.debug:
    CFG.epochs = 1
    CFG.print_freq = 10
    CFG.eval_freq = 45

import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer


from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


print('load data')
"""
train = pd.read_csv('input/train.csv')
train['essay_text']  = train['essay_id'].progress_apply(lambda x: get_essay(x, is_train=True))

train['discourse_text'] = train['discourse_text'].progress_apply(lambda x : resolve_encodings_and_normalize(x))
train['essay_text'] = train['essay_text'].progress_apply(lambda x : resolve_encodings_and_normalize(x))

SEP = tokenizer.sep_token
train['text'] = train['discourse_type'] + ' ' + train['discourse_text'] + SEP + train['essay_text']
print(train.head())

skf = StratifiedGroupKFold(n_splits=CFG.n_splits) # , shuffle=True, random_state=CFG.seed)
train['fold'] = -1
train['label'] = train['discourse_effectiveness'].map({'Ineffective':0, 'Adequate':1, 'Effective':2})
for i, (_, val_) in enumerate(skf.split(train, train['label'], groups=train['essay_id'])):
    train.loc[val_, 'fold'] = int(i)
train.to_csv('input/train_fold.csv', index=False)
"""
train = pd.read_csv('input/train_fold.csv')
print(train.fold.value_counts())

if CFG.debug:
    print(train.groupby('fold').size())
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)
    print(train.groupby('fold').size())


def get_sentence_features(train, col):
    train[col + '_num_chars'] = train[col].apply(len)
    train[col + '_num_capitals'] = train[col].apply(lambda x: sum(1 for c in x if c.isupper()))
    train[col + '_caps_vs_length'] = train.apply(lambda row: row[col + '_num_chars'] / (row[col + '_num_capitals']+1e-5), axis=1)
    train[col + '_num_exclamation_marks'] = train[col].apply(lambda x: x.count('!'))
    train[col + '_num_question_marks'] = train[col].apply(lambda x: x.count('?'))
    train[col + '_num_punctuation'] = train[col].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    train[col + '_num_symbols'] = train[col].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    train[col + '_num_words'] = train[col].apply(lambda x: len(x.split()))
    train[col + '_num_unique_words'] = train[col].apply(lambda comment: len(set(w for w in comment.split())))
    train[col + '_words_vs_unique'] = train[col + '_num_unique_words'] / train[col + '_num_words']
    return train


train = get_sentence_features(train, 'discourse_text')
train = get_sentence_features(train, 'essay_text')
train['div_discourse_text_num_chars_essay_text_num_chars'] = train['discourse_text_num_chars'] / train['essay_text_num_chars']
train['div_discourse_text_num_words_essay_text_num_words'] = train['discourse_text_num_words'] / train['essay_text_num_words']
train['div_discourse_text_num_unique_words_essay_text_num_unique_words'] = train['discourse_text_num_unique_words'] / train['essay_text_num_unique_words']


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df['label'].values
        self.text_features = df[CFG.text_features].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length = self.max_len
        )
        return {
            'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'target':self.targets[index],
            'text_features':self.text_features[index],
            }


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["text_features"] = [sample["text_features"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["text_features"] = torch.tensor(output["text_features"], dtype=torch.float)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=True)


def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings

    
class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
    
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


# ====================================================
# Model
# ====================================================
from torch.cuda.amp import autocast
class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()

        self.cfg = CFG
        self.config = AutoConfig.from_pretrained(model_name)
        #self.config.hidden_dropout_prob = 0
        #self.config.attention_probs_dropout_prob = 0
        #print(self.config)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        # self.pooler = MeanPooling()

        self.bilstm = nn.LSTM(self.config.hidden_size, (self.config.hidden_size) // 2, num_layers=2,
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

        self.linear_feature = nn.Sequential(
            nn.Linear(23, 128),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size + 128, self.cfg.target_size)
        )


        # Freeze
        if self.cfg.freezing:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:2])

        # Gradient Checkpointing
        #if self.cfg.gradient_checkpoint:
        #    self.model.gradient_checkpointing_enable() 

        if self.cfg.reinit_layers > 0:
            layers = self.model.encoder.layer[-self.cfg.reinit_layers:]
            for layer in layers:
                for module in layer.modules():
                    self._init_weights(module)

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        mll = log_loss(
            targets.cpu().detach().numpy(),
            softmax(outputs.cpu().detach().numpy()),
            labels=[0, 1, 2],
        )
        return mll

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, text_features, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        # LSTM/GRU header
#         all_hidden_states = torch.stack(transformer_out[1])
#         sequence_output = self.pooler(all_hidden_states)

        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        linear_feature = self.linear_feature(text_features)

        sequence_output = torch.cat([sequence_output, linear_feature], 1)

        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        #if targets is not None:
        #    metric = self.monitor_metrics(logits, targets)
        #    return logits, metric

        return logits#, 0.


def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

def train_one_epoch(model, optimizer, scheduler, dataloader, valid_loader, device, epoch, best_score, valid_labels):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)

    dataset_size = 0
    running_loss = 0

    start = end = time.time()

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        text_features = data['text_features'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        with autocast(enabled=CFG.apex):
            outputs = model(ids, mask, text_features)
            loss = criterion(outputs, targets)

        #accumulate
        loss = loss / CFG.n_accumulate
        scaler.scale(loss).backward()
        if (step +1) % CFG.n_accumulate == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  .format(epoch+1, step, len(dataloader),
                          remain=timeSince(start, float(step+1)/len(dataloader))))


        if (step > 0) & (step % CFG.eval_freq == 0) :

            valid_epoch_loss, pred = valid_one_epoch(model, valid_loader, device, epoch)

            score = get_score(pred, valid_labels)

            LOGGER.info(f'Epoch {epoch+1} Step {step} - avg_train_loss: {epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}')
            LOGGER.info(f'Epoch {epoch+1} Step {step} - Score: {score:.4f}')

            if score < best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} Step {step} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': pred},
                            OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    gc.collect()

    return epoch_loss, valid_epoch_loss, pred, best_score


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0

    start = end = time.time()
    pred = []

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        text_features = data['text_features'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)
        outputs = model(ids, mask, text_features)
        loss = criterion(outputs, targets)
        pred.append(outputs.to('cpu').numpy())

        running_loss += (loss.item()* batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  .format(step, len(dataloader),
                          remain=timeSince(start, float(step+1)/len(dataloader))))

    pred = np.concatenate(pred)

    return epoch_loss, pred


def train_loop(fold):
    #wandb.watch(model, log_freq=100)

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.fold != fold].reset_index(drop=True)
    valid_data = train[train.fold == fold].reset_index(drop=True)
    valid_labels = valid_data.label.values

    trainDataset = FeedBackDataset(train_data, CFG.tokenizer, CFG.max_len)
    validDataset = FeedBackDataset(valid_data, CFG.tokenizer, CFG.max_len)

    train_loader = DataLoader(trainDataset,
                              batch_size = CFG.batch_size,
                              shuffle=True,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=True)

    valid_loader = DataLoader(validDataset,
                              batch_size = CFG.batch_size * 2,
                              shuffle=False,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=False)

    #if CFG.itpt_path:
    #    model = FeedBackModel(CFG.itpt_path)
    #    print('load itpt model')
    #else:
    model = FeedBackModel(CFG.model)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)
    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.epochs):
        if epoch == (CFG.epochs - 2):
            break

        start_time = time.time()

        train_epoch_loss, valid_epoch_loss, pred, best_score = train_one_epoch(model, optimizer, scheduler, train_loader, valid_loader, device, epoch, best_score, valid_labels)
        # valid_epoch_loss, pred = valid_one_epoch(model, valid_loader, device, epoch)

        score = get_score(pred, valid_labels)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': pred},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_data['pred_0'] = predictions[:, 0]
    valid_data['pred_1'] = predictions[:, 1]
    valid_data['pred_2'] = predictions[:, 2]


    torch.cuda.empty_cache()
    gc.collect()

    return valid_data


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df['label'].values
        preds = oof_df[['pred_0', 'pred_1', 'pred_2']].values.tolist()
        score = get_score(preds, labels)
        LOGGER.info(f'Score: {score:<.4f}')

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
        oof_df.to_csv(OUTPUT_DIR+f'oof_df.csv', index=False)



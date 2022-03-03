import io
import os
import json
import numpy as np
import csv
import argparse
import random
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer

# DataSet
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def _loading_dataset(hparams):
    dataset = []
    column = hparams.column
    target_column = hparams.target_column
    if target_column == -1:
        for file in hparams.files:
            logging.info(f'loading {file}')
            if file.endswith('.csv') or file.endswith('.tsv'):
                sep = ',' if file.endswith('.csv') else '\t'
                with io.open(file, encoding=hparams.encoding) as f:
                    reader = csv.reader(f, delimiter=sep)
                    for row in reader:
                        dataset.append(row[column])
            elif file.endswith('.jsonl'):
                with io.open(file, encoding=hparams.encoding) as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        dataset.append(data[column])
            else:
                with io.open(file, encoding=hparams.encoding) as f:
                    for line in f.readlines():
                        dataset.append(line.rstrip('\n'))
    else:
        for file in hparams.files:
            logging.info(f'loading {file}')
            if file.endswith('.csv') or file.endswith('.tsv'):
                sep = ',' if file.endswith('.csv') else '\t'
                with io.open(file, encoding=hparams.encoding) as f:
                    reader = csv.reader(f, delimiter=sep)
                    for row in reader:
                        dataset.append((row[column], row[target_column]))
            elif file.endswith('.jsonl'):
                with io.open(file, encoding=hparams.encoding) as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        dataset.append((data[column], data[target_column]))
            else:
                with io.open(file, encoding=hparams.encoding) as f:
                    for line in f.readlines():
                        d = line.rstrip('\n')
                        dataset.append((d, d))
    logging.info(f'loaded {len(dataset)} dataset')
    return dataset

class TSVDataset(Dataset):
    def __init__(self, hparams, dataset=None):
        self.hparams = hparams
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = self._loading_dataset(self.hparams)
        self.encode = hparams.encode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.encode(self.dataset[index])

    def clone(self, new_dataset):
        return self.__class__(self.hparams, new_dataset)

    def random_split(self, train_size=0.7, test_size=0):
        random.shuffle(self.dataset)
        size = len(self.dataset)
        test_size = self._ratio_size(test_size, size)
        if test_size < size:
            size -= test_size
        else:
            test_size = 0
        train_size = self._ratio_size(train_size, size)
        trainset = self.dataset[:train_size]
        validset = self.dataset[train_size:size]
        if test_size > 0:
            testset = self.dataset[size:]
            return self.clone(trainset), self.clone(validset), self.clone(testset)
        return self.clone(trainset), self.clone(validset)

    def _ratio_size(self, ratio, len):
        if ratio <= 1.0:
            return int(len * ratio)
        return ratio

MULTITASKING_TRANSFORM = {

}

def transform_nmt(pair):
    src, tgt = pair
    prefix, sep, body = src.partition(':')
    if prefix in MULTITASKING_TRANSFORM:
        src, tgt = MULTITASKING_TRANSFORM[prefix](body, tgt)
        return src, tgt
    return pair

def encode_nmt(pair, hparams):
    src, tgt = hparams.transform(pair)
    inputs = hparams.tokenizer.batch_encode_plus(
        [src], max_length=hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
    targets = hparams.tokenizer.batch_encode_plus(
        [tgt], max_length=hparams.hparams.target_max_seq_length, truncation=True, padding="max_length", return_tensors="pt")

    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()

    target_ids = targets["input_ids"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {"source_ids": source_ids, "source_mask": source_mask,
            "target_ids": target_ids, "target_mask": target_mask}

# denosing objective

EOS = 1
ID = 250099
s
def _masking_pad(input_ids, masked, max_length=128):
    c = 0
    prev_index = None
    for index in masked:
        if prev_index == index - 1:
            input_ids[index] = None
        else:
            input_ids[index] = ID - c
            c += 1
        prev_index = index
    s = [ids for ids in input_ids if ids != None] + [EOS]
    pad = [0] * (max_length-len(s))
    return s + pad, [1] * len(s) + pad

debug_count_masking = 0

def encode_denoising_objective(line, hparams):
    global debug_count_masking
    inputs = hparams.tokenizer.batch_encode_plus(
        [line], max_length=hparams.max_seq_length, truncation=True, return_tensors="pt")
    # print(inputs)
    input_ids = inputs["input_ids"].squeeze().tolist()
    input_ids = input_ids[:-1]  # </s> を除いたinputs のlist
    n_tokens = len(input_ids)   # 字句数
    n = max(int((n_tokens / 2) * hparams.masking_ratio), 1)
    input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
    source, source_attn = _masking_pad(
        input_ids[:], input_masked, hparams.max_seq_length)
    output_masked = list(
        set(list(range(0, n_tokens))) - set(input_masked))
    target, target_attn = _masking_pad(
        input_ids[:], output_masked, hparams.max_seq_length)
    return {"source_ids": torch.tensor(source), "source_mask": torch.tensor(source_attn),
            "target_ids": torch.tensor(target), "target_mask": torch.tensor(target_attn)}

def encode_bert_style(line, hparams):
    global debug_count_masking
    inputs = hparams.tokenizer.batch_encode_plus(
        [line], max_length=hparams.max_seq_length, truncation=True, return_tensors="pt")
    # print(inputs)
    input_ids = inputs["input_ids"].squeeze().tolist()
    input_ids = input_ids[:-1]  # </s> を除いたinputs のlist
    n_tokens = len(input_ids)   # 字句数
    n = max(int((n_tokens / 2) * hparams.masking_ratio), 1)
    input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
    source, source_attn = _masking_pad(
        input_ids[:], input_masked, hparams.max_seq_length)
    targets = hparams.tokenizer.batch_encode_plus(
        [line], max_length=hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
    return {"source_ids": torch.tensor(source), "source_mask": torch.tensor(source_attn),
            "target_ids": targets["input_ids"].squeeze(), "target_mask": targets["attention_mask"].squeeze()}

# argparse

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _add_arguments(parser, args_dict):
    for key in args_dict:
        option_name = f'--{key}'
        default = args_dict[key]
        if isinstance(default, bool):
            if default == False:
                parser.add_argument(
                    option_name, action='store_true', default=default)
            elif default == True:
                parser.add_argument(
                    option_name, action='store_false', default=default)
        elif isinstance(default, int):
            parser.add_argument(option_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(option_name, type=float, default=default)
        elif isinstance(default, str):
            parser.add_argument(option_name, default=default)


def init_hparams(init_dict, description='Trainer of mT5 on ABCI'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', nargs='+', help='files')
    _add_arguments(parser, init_dict)
    parser.add_argument('-q', '--quantize', action='store_true',
                        help='quantize model')
    hparams = parser.parse_args()

    _set_seed(hparams.seed)

    if not os.path.isdir(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    if hparams.additional_special_tokens == '':
        hparams.additional_special_tokens = None
    else:
        hparams.additional_special_tokens = hparams.additional_special_tokens.split()

    hparams.tokenizer = MT5Tokenizer.from_pretrained(
        hparams.tokenizer_name_or_path, is_fast=True)
    if not hparams.additional_special_tokens:
        hparams.tokenizer.add_tokens(hparams.additional_special_tokens)

    return hparams

def _main():
    init_dict = dict(
        model_name_or_path='google/mt5-small',
        tokenizer_name_or_path='google/mt5-small',
        additional_special_tokens='',
        seed=42,
        encoding='utf_8',
        column=0, target_column=-1,
        k_fold=5,  # cross validation
        max_seq_length=128,
        target_max_seq_length=128,
        # unsupervised training option
        mlm=False,
        masking_ratio=0.15,
        bert_style=False,
    )
    hparams = init_hparams(init_dict)
    dataset = TSVDataset(hparams)

if __name__ == '__main__':
    _main()

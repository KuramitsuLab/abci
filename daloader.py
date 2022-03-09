from multiese2_da import transform_multiese
import io
import os
import json
import numpy as np
import csv
import argparse
import random
import logging

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")


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
                        if column < len(row):
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
                        if column < len(row) and target_column < len(row):
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


class DADataset(Dataset):
    def __init__(self, hparams, dataset=None):
        self.hparams = hparams
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = _loading_dataset(self.hparams)
        self.encode = hparams.encode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.encode(self.dataset[index], self.hparams)

    def clone(self, new_dataset):
        return self.__class__(self.hparams, new_dataset)

    # def random_split(self, train_size=0.7, test_size=0):
    #     random.shuffle(self.dataset)
    #     size = len(self.dataset)
    #     test_size = self._ratio_size(test_size, size)
    #     if test_size < size:
    #         size -= test_size
    #     else:
    #         test_size = 0
    #     train_size = self._ratio_size(train_size, size)
    #     trainset = self.dataset[:train_size]
    #     validset = self.dataset[train_size:size]
    #     if test_size > 0:
    #         testset = self.dataset[size:]
    #         return self.clone(trainset), self.clone(validset), self.clone(testset)
    #     return self.clone(trainset), self.clone(validset)

    # def _ratio_size(self, ratio, len):
    #     if ratio <= 1.0:
    #         return int(len * ratio)
    #     return ratio


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class KFoldDataset(Dataset):
    def __init__(self, dataset: Dataset, kfold=5):
        self.allset = dataset
        self.trainset = Subset(dataset, [])
        self.validset = Subset(dataset, [])
        self.kfold = 5
        self.index = self.kfold

    def __getitem__(self, idx):
        return self.allset[idx]

    def __len__(self):
        return len(self.allset)

    def split(self):
        self.index += 1
        kfold = self.kfold
        index = self.index % kfold
        train_index = []
        valid_index = []
        for i in range(len(self.allset)):
            if i % kfold == index:
                valid_index.append(i)
            else:
                train_index.append(i)
        random.shuffle(train_index)
        random.shuffle(valid_index)
        self.trainset.indices = train_index
        self.validset.indices = valid_index
        return self.trainset, self.validset


# MULTITASKING_TRANSFORM


def transform_nmt(pair, hparams):
    return pair


def encode_t5(pair, hparams):
    src, tgt = hparams.transform(pair, hparams)
    inputs = hparams.tokenizer.batch_encode_plus(
        [src], max_length=hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
    targets = hparams.tokenizer.batch_encode_plus(
        [tgt], max_length=hparams.target_max_seq_length, truncation=True, padding="max_length", return_tensors="pt")

    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()

    target_ids = targets["input_ids"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {"source_ids": source_ids, "source_mask": source_mask,
            "target_ids": target_ids, "target_mask": target_mask}


def encode_string(pair, hparams):
    src, tgt = hparams.transform(pair, hparams)
    return src, tgt


# extra_id
# denosing objective

EOS = 1
ID = 250099
EXTRA_ID = {
}


def _setup_extra_id(hparams):
    if '<extra_id_0>' not in hparams.tokenizer.vocab:
        hparams.tokenizer.add_tokens(
            [f'<extra_id_{i}>' for i in range(100)])
        hparams.vocab_size += 100
    for c in range(100):
        EXTRA_ID[c] = hparams.tokenizer.vocab[f'<extra_id_{c}>']


def _extra_id(tokenizer, c):
    return EXTRA_ID[c]  # 250099 - c


def _masking_pad(input_ids, masked, max_length=128):
    c = 0
    prev_index = None
    for index in masked:
        if prev_index == index - 1:
            input_ids[index] = None
        else:
            input_ids[index] = _extra_id(c)  # ID - c
            c += 1
        prev_index = index
    s = [ids for ids in input_ids if ids != None] + [EOS]
    pad = [0] * (max_length-len(s))
    return s + pad, [1] * len(s) + pad


def encode_denoising_objective(line, hparams):
    inputs = hparams.tokenizer.batch_encode_plus(
        [line], max_length=hparams.max_seq_length, truncation=True, return_tensors="pt")
    print(inputs)
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


def init_hparams(init_dict, description='Trainer of mT5 on ABCI', Tokenizer=None, transform=transform_multiese):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', nargs='+', help='files')
    _add_arguments(parser, init_dict)
    # parser.add_argument('-q', '--quantize', action='store_true',
    #                     help='quantize model')
    hparams = parser.parse_args()
    _set_seed(hparams.seed)

    if not os.path.isdir(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    if hparams.additional_tokens == '':
        hparams.additional_special_tokens = None
    else:
        hparams.additional_tokens = hparams.additional_tokens.split()

    if Tokenizer is None:
        hparams.encode = encode_string
        if hparams.mlm:
            hparams.target_column = -1
    else:
        hparams.tokenizer = Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, is_fast=True)
        hparams.vocab_size = hparams.tokenizer.vocab_size
        if not hparams.additional_tokens:
            hparams.tokenizer.add_tokens(hparams.additional_tokens)
            hparams.vocab_size += len(hparams.additional_tokens)
        if hparams.mlm:
            hparams.encode = encode_denoising_objective
            hparams.target_column = -1
            _setup_extra_id(hparams)
        else:
            hparams.encode = encode_t5

    hparams.transform = transform
    if not hasattr(hparams, 'da_choice'):
        hparams.da_choice = 0.1
    if not hasattr(hparams, 'da_shuffle'):
        hparams.da_shuffle = 0.3
    return hparams


def _main():
    init_dict = dict(
        output_dir='./model',  # path to save the checkpoints
        model_name_or_path='google/mt5-small',
        tokenizer_name_or_path='google/mt5-small',
        additional_tokens='<e0> <e1> <e2> <e3> <e4> <e5> <e6> <e7> <e8> <e9>',
        seed=42,
        encoding='utf_8',
        column=0, target_column=1,
        kfold=5,  # cross validation
        max_seq_length=128,
        target_max_seq_length=128,
        # da
        da_choice=0.1, da_shuffle=0.3,
        # unsupervised training option
        mlm=False,
        masking_ratio=0.15,
        bert_style=False,
    )
    hparams = init_hparams(init_dict, transform=transform_multiese)
    print(hparams)
    dataset = KFoldDataset(DADataset(hparams))
    train, valid = dataset.split()
    for i in range(min(10, len(train))):
        print(train[i])


if __name__ == '__main__':
    _main()

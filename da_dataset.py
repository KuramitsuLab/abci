import io
import os
import json
import sys
import numpy as np
import csv
import argparse
import random
import logging
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from da_multiese import transform_multitask
from da_masking import get_transform_masking


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
                            src = row[column]
                            tgt = row[target_column]
                            _append_dup(hparams, dataset, src, tgt)
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


def _append_da(dataset, src, tgt):
    dataset.append((src, tgt))
    brace = src.count('{')
    square = src.count('[')
    vbar = src.count('|')
    if brace > 0 and vbar != 0:
        dataset.append((src, tgt))
    if vbar > 5 or square > 3:
        dataset.append((src, tgt))


def _append_dup(hparams, dataset, src, tgt):
    _append_da(dataset, src, tgt)
    if src.startswith('trans:'):
        src = src.replace('trans:', 'code:')
        _append_da(dataset, src, tgt)


def transform_nop(pair, hparams):
    if isinstance(pair, str):
        return pair, pair
    return pair


class DADataset(Dataset):
    def __init__(self, hparams, dataset=None):
        self.hparams = hparams
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = _loading_dataset(self.hparams)
        if hparams.masking:
            self.transform = get_transform_masking(hparams)
        else:
            self.transform = transform_multitask
        self.encode = hparams.encode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src, tgt = self.transform(self.dataset[index], self.hparams)
        return self.encode(src, tgt, self.hparams)

    def test_and_save(self, testing_fn, transform=None, file=sys.stdout):
        encode_orig = self.encode
        transform_orig = self.transform
        if transform is not None:
            self.transform = transform
        self.encode = encode_string
        try:
            if self.hparams.progress_bar:
                for src, tgt in tqdm(self):
                    src, gen, tgt = testing_fn(src, tgt)
                    print(f'{src}\t{gen}\t{tgt}', file=file)
            else:
                for src, tgt in self:
                    src, gen, tgt = testing_fn(src, tgt)
                    print(f'{src}\t{gen}\t{tgt}', file=file)
        finally:
            self.encode = encode_orig
            self.transform = transform_orig


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

    def test_and_save(self, gen_fn=lambda src, tgt: (src, tgt, None), file=sys.stdout):
        if isinstance(self.allset, DADataset):
            if isinstance(file, str):
                with open(file, 'w') as f:
                    self.allset.test_and_save(gen_fn, file=f)
            else:
                self.allset.test_and_save(gen_fn, file=file)


# MULTITASKING_TRANSFORM


def encode_t5(src, tgt, hparams):
    inputs = hparams.tokenizer.batch_encode_plus(
        [src],
        max_length=hparams.max_seq_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")
    targets = hparams.tokenizer.batch_encode_plus(
        [tgt],
        max_length=hparams.target_max_seq_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")

    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()

    target_ids = targets["input_ids"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {
        "source_ids": source_ids.to(dtype=torch.long),
        "source_mask": source_mask.to(dtype=torch.long),
        "target_ids": target_ids.to(dtype=torch.long),
        "target_mask": target_mask.to(dtype=torch.long),
    }


def encode_string(src, tgt, hparams):
    return src, tgt


def _setup_extra_id(hparams):
    if '<extra_id_0>' not in hparams.tokenizer.vocab:
        hparams.tokenizer.add_tokens(
            [f'<extra_id_{i}>' for i in range(100)])
        hparams.vocab_size += 100

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


def init_hparams(init_dict, description='Trainer of mT5 on ABCI', Tokenizer=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', nargs='+', help='files')
    parser.add_argument('--name', type=str, default='', help='project name')
    _add_arguments(parser, init_dict)
    # parser.add_argument('-q', '--quantize', action='store_true',
    #                     help='quantize model')
    hparams = parser.parse_args()

    if hparams.name == '':
        hparams.suffix = ''
        hparams.prefix = ''
    else:
        hparams.prefix = hparams.name
        hparams.suffix = f'_{hparams.name}'
        hparams.output_dir = f'{hparams.prefix}/{hparams.output_dir}'

    _set_seed(hparams.seed)

    if not os.path.isdir(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    if hparams.masking or hparams.target_column == -1:
        hparams.masking = True
        hparams.target_column = -1

    if hparams.additional_tokens == '':
        hparams.additional_special_tokens = None
    else:
        hparams.additional_tokens = hparams.additional_tokens.split()

    if Tokenizer is None:
        hparams.encode = encode_string
    else:
        if not hasattr(hparams, 'use_fast_tokenizer'):
            hparams.use_fast_tokenizer = False
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        hparams.tokenizer = Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, is_fast=hparams.use_fast_tokenizer)
        hparams.vocab_size = hparams.tokenizer.vocab_size
        if hparams.additional_tokens:
            hparams.tokenizer.add_tokens(hparams.additional_tokens)
            hparams.vocab_size += len(hparams.additional_tokens)
        if hparams.masking:
            _setup_extra_id(hparams)
        hparams.encode = encode_t5
    hparams.data_duplication = True
    # if not hasattr(hparams, 'da_choice'):
    #     hparams.da_choice = 0.5
    # if not hasattr(hparams, 'da_shuffle'):
    #     hparams.da_shuffle = 0.5
    _setup_logger(hparams)
    return hparams


def _setup_logger(hparams):
    log_file = f'log{hparams.suffix}.txt'

    # ストリームハンドラの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))

    # ファイルハンドラの設定
    file_handler = FileHandler(log_file)

    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(
        Formatter(
            "%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
    )
    # ルートロガーの設定
    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    logging.info(f'PyTorch: {torch.__version__}')
    logging.info(f'hparams: {hparams}')


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
        progress_bar=False,
        # da
        da_choice=1.0, da_shuffle=1.0, bos_token='',
        # unsupervised training option
        masking=False,
        masking_ratio=0.35,
        masking_style='denoising_objective',
    )
    hparams = init_hparams(init_dict)
    print(hparams)
    dataset = KFoldDataset(DADataset(hparams))
    dataset.test_and_save(gen_fn=lambda src, tgt: (src, tgt, ''))


if __name__ == '__main__':
    _main()

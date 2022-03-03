
# インポート
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter
import io
import os
import math
import numpy as np
import csv
import argparse
import random

import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    get_linear_schedule_with_warmup
)
import json
import logging

# from slackweb import Slack

# GPU利用有無
USE_GPU = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()

FILE_SUFFIX = f'{datetime.datetime.now():%Y%m%dT%H%M%S}'


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


args_dict = dict(
    # data_dir='./',  # path for data files
    output_dir='./mymodel',  # path to save the checkpoints
    output_suffix=FILE_SUFFIX,
    model_name_or_path='google/mt5-small',
    tokenizer_name_or_path='google/mt5-small',
    additional_special_tokens='',
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=1,  # 16
    n_gpu=N_GPU if USE_GPU else 0,
    early_stop_callback=False,
    # if you want to enable 16-bit training then install apex and set this to true
    fp_16=True if USE_GPU else False,
    opt_level='O2',
    max_grad_norm=1.0,
    seed=42,
    # data loader
    encoding='utf8',
    column=0, target_column=1,
    k_fold=5,  # cross validation
    max_seq_length=128,
    target_max_seq_length=128,
    # unsupervised training option
    mlm=False,
    masking_ratio=0.15,
    bert_style=False,
)


class MT5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # 事前学習済みモデルの読み込み
        self.model = MT5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)

        # トークナイザーの読み込み
        self.tokenizer = MT5Tokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path, is_fast=True)
        if not self.hparams.additional_special_tokens:
            self.tokenizer.add_tokens(self.hparams.additional_special_tokens)
        self.dataset = None
        self.train_dataset = None

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        #self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        #self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """バリデーション完了処理"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        #logging.info(f'loss {avg_loss} PPL {math.exp(avg_loss)}')
        self.update_kfold()

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     """テスト完了処理"""
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self):
        """データセットを作成する"""
        if self.hparams.mlm:
            return MaskedDataset(self.hparams, tokenizer=self.tokenizer)
        return T5Dataset(self.hparams, tokenizer=self.tokenizer)

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            if self.dataset is None:
                self.dataset = self.get_dataset()
            if self.hparams.k_fold > 2:
                valid_size = 1 / self.hparams.k_fold
                self.train_dataset = None
            else:
                valid_size = 0.15
            if self.train_dataset is None:
                train_index, valid_index = train_test_split(
                    range(len(self.dataset)), test_size=valid_size, random_state=hparams.seed)
                self.train_dataset = Subset(self.dataset, train_index)
                self.valid_dataset = Subset(self.dataset, valid_index)
            self.t_total = (
                (len(self.train_dataset) //
                 (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def update_kfold(self):
        if self.hparams.k_fold > 2:
            valid_size = 1 / self.hparams.k_fold
            train_index, valid_index = train_test_split(
                range(len(self.dataset)), test_size=valid_size, random_state=None)
            self.train_dataset.indices = train_index
            self.valid_dataset.indices = valid_index
            # logging.info(
            #     f'{self.hparams.k_fold}-fold cross validation: {len(train_index)} {valid_index}')

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        logging.info('loading train data loader')
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.eval_batch_size,
                          num_workers=4)


# DataSet
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class T5Dataset(Dataset):
    def __init__(self, hparams, dataset=None, tokenizer=None):
        self.hparams = hparams
        self.dataset = dataset
        self.tokenizer = tokenizer
        if dataset is None:
            self._loading_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def transform(self, pair):
        src, tgt = pair
        inputs = self.tokenizer.batch_encode_plus(
            [src], max_length=self.hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus(
            [tgt], max_length=self.hparams.target_max_seq_length, truncation=True, padding="max_length", return_tensors="pt")

        source_ids = inputs["input_ids"].squeeze()
        source_mask = inputs["attention_mask"].squeeze()

        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _loading_dataset(self):
        dataset = []
        column = hparams.column
        target_column = hparams.target_column
        for file in self.hparams.files:
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
        self.dataset = dataset

    def clone(self, new_dataset):
        return self.__class__(self.hparams, new_dataset, self.tokenizer)

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


def transform_nop(x):
    return x.strip()


debug_count_masking = 0


class MaskedDataset(T5Dataset):

    def transform(self, line):
        global debug_count_masking
        inputs = self.tokenizer.batch_encode_plus(
            [line], max_length=self.hparams.max_seq_length, truncation=True, return_tensors="pt")
        # print(inputs)
        input_ids = inputs["input_ids"].squeeze().tolist()
        input_ids = input_ids[:-1]  # </s> を除いたinputs のlist
        n_tokens = len(input_ids)   # 字句数
        n = max(int((n_tokens / 2) * self.hparams.masking_ratio), 1)
        input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
        source, source_attn = masking_pad(
            input_ids[:], input_masked, self.hparams.max_seq_length)
        f = torch.tensor
        if self.hparams.bert_style:
            targets = self.tokenizer.batch_encode_plus(
                [line], max_length=self.hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
            return {"source_ids": f(source), "source_mask": f(source_attn),
                    "target_ids": targets["input_ids"].squeeze(), "target_mask": targets["attention_mask"].squeeze()}
        else:
            output_masked = list(
                set(list(range(0, n_tokens))) - set(input_masked))
            target, target_attn = masking_pad(
                input_ids[:], output_masked, self.hparams.max_seq_length)
            return {"source_ids": f(source), "source_mask": f(source_attn),
                    "target_ids": f(target), "target_mask": f(target_attn)}

    def _loading_dataset(self):
        dataset = []
        column = hparams.column
        for file in self.hparams.files:
            logging.info(f'loading {file}')
            if file.endswith('.csv') or file.endswith('.tsv'):
                sep = ',' if file.endswith('.csv') else '\t'
                with io.open(file, encoding=hparams.encoding) as f:
                    reader = csv.reader(f, delimiter=sep)
                    for row in reader:
                        dataset.append(row[column])
            elif file.endswith('.jsonl'):
                with io.open(file, ) as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        dataset.append(data[column])
            else:
                with io.open(file) as f:
                    for line in f.readlines():
                        dataset.append(line.rstrip('\n'))
        logging.info(f'loaded {len(dataset)} dataset')
        self.dataset = dataset


EOS = 1
ID = 250099


def masking_pad(input_ids, masked, max_length=128):
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


def quantize_transform(model):
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model


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


def _setup_hparams():
    parser = argparse.ArgumentParser(description='Trainer of mT5 on ABCI')
    parser.add_argument('files', nargs='+', help='files')
    _add_arguments(parser, args_dict)

    # parser.add_argument('-e', '--epochs', help='epochs',
    #                     type=int, default=50)
    # parser.add_argument('--zip', action='store_true',
    #                     help='Save the model as a zip file')
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

    return hparams


def _setup_logger(hparams):
    log_file = os.path.join(
        hparams.output_dir, f'training_{FILE_SUFFIX}_log.txt')

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


def _main(hparams):
    _set_seed(hparams.seed)

    # logging.info(f'Start trainig: {hparams.start_date}')
    logging.info(f'Base model: {hparams.model_name_or_path} {hparams.files}')

    train_params = dict(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        gpus=hparams.n_gpu,
        max_epochs=hparams.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if hparams.fp_16 else 32,
        # amp_level=hparams.opt_level,
        gradient_clip_val=hparams.max_grad_norm,
        #    checkpoint_callback=checkpoint_callback,
        # callbacks=[LoggingCallback()],
    )

    model = MT5FineTuner(hparams)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # 最終エポックのモデルを保存
    tokenizer = model.tokenizer
    model = model.model
    if hparams.quantize:
        logging.info('Enabled quantization model')
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    tokenizer.save_pretrained(hparams.output_dir)
    model.save_pretrained(hparams.output_dir)


if __name__ == '__main__':
    hparams = _setup_hparams()
    _setup_logger(hparams)
    _main(hparams)


# インポート
from daloader import init_hparams
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter
import numpy as np
import random

import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import logging

from daloader import DADataset, KFoldDataset

# GPU利用有無
USE_GPU = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()

FILE_SUFFIX = f'{datetime.datetime.now():%Y%m%dT%H%M%S}'


args_dict = dict(
    # data_dir='./',  # path for data files
    output_dir='./model',  # path to save the checkpoints
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
    encoding='utf_8',
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
        self.tokenizer = self.hparams.tokenizer
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
        print('@@', 'cross validation')
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
        return KFoldDataset(DADataset(self.hparams))

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        print('@@', stage)
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                self.dataset = self.get_dataset()
                self.train_dataset, self.valid_dataset = self.dataset.split()
            self.t_total = (
                (len(self.train_dataset) //
                 (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def update_kfold(self):
        self.dataset.split()
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
        # training
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
    )
    hparams = init_hparams(init_dict, Tokenizer=AutoTokenizer)
    print(hparams)

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
    # if hparams.quantize:
    #     logging.info('Enabled quantization model')
    #     model = torch.quantization.quantize_dynamic(
    #         model, {torch.nn.Linear}, dtype=torch.qint8
    #     )
    tokenizer.save_pretrained(hparams.output_dir)
    model.save_pretrained(hparams.output_dir)


if __name__ == '__main__':
    _main()

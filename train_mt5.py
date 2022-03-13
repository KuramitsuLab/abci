import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    MT5ForConditionalGeneration, T5ForConditionalGeneration,
    AutoConfig, AutoModel, AutoTokenizer,
    get_linear_schedule_with_warmup
)
import os
import logging

from da_dataset import init_hparams, DADataset, KFoldDataset

# GPU利用有無
USE_GPU = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()


class MT5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # 事前学習済みモデルの読み込み
        if '/mt5' in self.hparams.model_name_or_path:
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path)
        # config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        # self.model = AutoModel.from_config(config)
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
        self.log("val_loss", avg_loss, prog_bar=self.hparams.progress_bar)
        #logging.info(f'loss {avg_loss} PPL {math.exp(avg_loss)}')
        #print('@@', 'cross validation')
        self.dataset.split()
        if self.hparams.save_checkpoint:
            print(f'saving checkpoint model to {self.hparams.output_dir}')
            self.tokenizer.save_pretrained(self.hparams.output_dir)
            self.model.save_pretrained(self.hparams.output_dir)

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
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                self.dataset = self.get_dataset()
                self.train_dataset, self.valid_dataset = self.dataset.split()
            self.t_total = (
                (len(self.train_dataset) //
                 (self.hparams.batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.max_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        #logging.info('loading train data loader')
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          drop_last=True, shuffle=True,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


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
        masking=False,
        masking_ratio=0.35,
        masking_style='denoising',
        # training
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        batch_size=8,
        num_workers=2,  # os.cpu_count(),
        # train_batch_size=8,
        save_checkpoint=False,
        progress_bar=True,
        # eval_batch_size=8,
        max_epochs=50,
        limit_batches=-1,
        gradient_accumulation_steps=1,  # 16
        n_gpu=N_GPU if USE_GPU else 0,
        early_stop_callback=True,
        # if you want to enable 16-bit training then install apex and set this to true
        fp_16=False,
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
        max_epochs=hparams.max_epochs,
        # early_stop_callback=False,
        precision=16 if hparams.fp_16 else 32,
        # amp_level=hparams.opt_level,
        gradient_clip_val=hparams.max_grad_norm,
        #    checkpoint_callback=checkpoint_callback,
        # callbacks=[LoggingCallback()],
        callbacks=[EarlyStopping(monitor="val_loss")],
        # turn off automatic checkpointing
        enable_checkpointing=False,
        enable_progress_bar=hparams.progress_bar,
        # run batch size scaling, result overrides hparams.batch_size
        auto_scale_batch_size="binsearch" if hparams.batch_size <= 2 else None,
        # run learning rate finder, results override hparams.learning_rate
        # auto_lr_find=True,
        devices="auto", accelerator="auto",
        limit_train_batches=1.0 if hparams.limit_batches == -1 else hparams.limit_batches,
        limit_val_batches=1.0 if hparams.limit_batches == -1 else hparams.limit_batches//4,
    )

    model = MT5FineTuner(hparams)
    trainer = pl.Trainer(**train_params)
    trainer.tune(model)
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

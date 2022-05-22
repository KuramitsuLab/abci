import logging
import torch
import torch.nn as nn

from timeit import default_timer as timer
from train_common import parse_hparams, load_TrainTestDataSet

from transformers import AutoTokenizer
from pytorch_t import (
    Seq2SeqTransformer,
    get_transform, train, evaluate,
    save_model, load_pretrained, load_nmt,
    PAD_IDX, DEVICE
)

# from morichan


def get_optimizer(hparams, model):
    # オプティマイザの定義 (Adam)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    return optimizer


def get_optimizer_adamw(hparams, model):
    # オプティマイザの定義 (AdamW)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=hparams.learning_rate,
                                  eps=hparams.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=hparams.warmup_steps,
    #     num_training_steps=t_total
    # )
    return optimizer


setup = dict(
    model_name_or_path='megagonlabs/t5-base-japanese-web',
    tokenizer_name_or_path='megagonlabs/t5-base-japanese-web',
    additional_tokens='<b> </b> <nl> <e0> <e1> <e2> <e3> <e4> <e5> <e6> <e7> <e8> <e9>',
    seed=42,
    encoding='utf_8',
    column=0, target_column=1,
    kfold=5,  # cross validation
    max_length=80,
    target_max_length=80,
    # training
    max_epochs=30,
    num_workers=2,  # os.cpu_count(),
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    # learning_rate=0.0001,
    # adam_epsilon=1e-9,
    # weight_decay=0
    # Transformer
    emb_size=512,  # BERT の次元に揃えれば良いよ
    nhead=8,
    fnn_hid_dim=512,  # 変える
    batch_size=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)


def _main():
    hparams = parse_hparams(setup, Tokenizer=AutoTokenizer)
    _, _, transform = get_transform(
        hparams.tokenizer, hparams.max_length, hparams.target_max_length)
    train_dataset, valid_dataset = load_TrainTestDataSet(
        hparams, transform=transform)

    if hparams.model_name_or_path.endswith('.pt'):
        model = load_pretrained(hparams.model_name_or_path, DEVICE)
    else:
        vocab_size = hparams.vocab_size
        model = Seq2SeqTransformer(hparams.num_encoder_layers, hparams.num_decoder_layers,
                                   hparams.emb_size, hparams.nhead,
                                   vocab_size+4, vocab_size+4, hparams.fnn_hid_dim)

    # TODO: ?
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('Parameter:', params)

    # デバイスの設定
    model = model.to(DEVICE)

    # 損失関数の定義 (クロスエントロピー)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # オプティマイザの定義 (Adam)
    optimizer = get_optimizer(hparams, model)

    train_list = []
    valid_list = []
    logging.info(f'start training max_epochs={hparams.max_epochs}')
    for epoch in range(1, hparams.max_epochs+1):
        start_time = timer()
        train_loss = train(train_dataset, model,
                           hparams.batch_size, loss_fn, optimizer)
        train_list.append(train_loss)
        end_time = timer()
        val_loss = evaluate(valid_dataset, model, hparams.batch_size, loss_fn)
        valid_list.append(val_loss)
        logging.info(
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))

    save_model(hparams, model,
               f'{hparams.output_dir}/tf_{hparams.project}.pt')

    print('Testing on ', DEVICE)
    train_dataset, valid_dataset = load_TrainTestDataSet(hparams)
    generate = load_nmt(
        f'{hparams.output_dir}/tf_{hparams.project}.pt', AutoTokenizer=AutoTokenizer)
    valid_dataset.test_and_save(
        generate, f'{hparams.output_dir}/result_test.tsv')
    train_dataset.test_and_save(
        generate, f'{hparams.output_dir}/result_train.tsv', max=1000)

# greedy search を使って翻訳結果 (シーケンス) を生成


if __name__ == '__main__':
    _main()

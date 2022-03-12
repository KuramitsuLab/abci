from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from da_dataset import init_hparams, DADataset, KFoldDataset
from pytorch_t import Seq2SeqTransformer, create_mask, save_model, load_pretrained, load_nmt

import torch
import torch.nn as nn

from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

print(torch.__version__)


# from morichan

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0

# モデルが予測を行う際に、未来の単語を見ないようにするためのマスク


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for b in batch:
        # print(b)
        src_batch.append(b["source_ids"])
        tgt_batch.append(b["target_ids"])
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train(hparams, train_iter, model, loss_fn, optimizer):
    model.train()
    losses = 0

    # 学習データ
    #collate_fn = string_collate(hparams)
    train_dataloader = DataLoader(
        train_iter, batch_size=hparams.batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(hparams, val_iter, model, loss_fn):
    model.eval()
    losses = 0

    #collate_fn = string_collate(hparams)
    val_dataloader = DataLoader(
        val_iter, batch_size=hparams.batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


setup = dict(
    output_dir='./model',  # path to save the checkpoints
    model_name_or_path='',
    tokenizer_name_or_path='sonoisa/t5-base-japanese',
    additional_tokens='<e0> <e1> <e2> <e3> <e4> <e5> <e6> <e7> <e8> <e9> encourage: translate: describe: option:',
    seed=42,
    encoding='utf_8',
    column=0, target_column=1,
    kfold=5,  # cross validation
    max_seq_length=128,
    target_max_seq_length=128,
    # da
    da_choice=0.5, da_shuffle=0.3,
    # unsupervised training option
    masking=False,
    masking_ratio=0.35,
    masking_style='denoising',
    # training
    max_epochs=50,
    num_workers=2,  # os.cpu_count(),
    learning_rate=0.0001,
    adam_epsilon=1e-9,
    # Transformer
    emb_size=512,  # BERT の次元に揃えれば良いよ
    nhead=8,
    fnn_hid_dim=512,  # 変える
    batch_size=64,
    num_encoder_layers=3,
    num_decoder_layers=3,
)


def _main():
    global PAD_IDX
    hparams = init_hparams(setup, Tokenizer=AutoTokenizer)
    print(hparams)
    dataset = KFoldDataset(DADataset(hparams))

    vocab_size = hparams.vocab_size
    PAD_IDX = hparams.tokenizer.pad_token_id
    if hparams.model_name_or_path.endswith('.pt'):
        model = load_pretrained(hparams.model_name_or_path, DEVICE)
    else:
        model = Seq2SeqTransformer(hparams.num_encoder_layers, hparams.num_decoder_layers,
                                   hparams.emb_size, hparams.nhead, vocab_size, vocab_size,
                                   hparams.fnn_hid_dim)

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        betas=(0.9, 0.98), eps=hparams.adam_epsilon
    )

    train_list = []
    valid_list = []

    for epoch in range(1, hparams.max_epochs+1):
        start_time = timer()
        train_iter, val_iter = dataset.split()
        train_loss = train(hparams, train_iter, model, loss_fn, optimizer)
        train_list.append(train_loss)
        end_time = timer()
        val_loss = evaluate(hparams, val_iter, model, loss_fn)
        valid_list.append(val_loss)
        print(
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))

    save_model(hparams, model, f'transformer_model{hparams.suffix}.pt')
    generate = load_nmt(f'transformer_model{hparams.suffix}.pt')
    def testing(src, tgt): return (src, generate(src), tgt)
    dataset.test_and_save(testing, file=f'result{hparams.suffix}.tsv')
    # for ins, ous in dataset:
    #     print(ins, ous, generate(ins))


# greedy search を使って翻訳結果 (シーケンス) を生成


if __name__ == '__main__':
    _main()

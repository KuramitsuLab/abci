from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from da_dataset import init_hparams, DADataset, KFoldDataset

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, List
#import sentencepiece as spm
import io
import math

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

print(torch.__version__)


# from morichan

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PAD_IDX = 0

# モデルが予測を行う際に、未来の単語を見ないようにするためのマスク


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# ソースとターゲットのパディングトークンを隠すためのマスク


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# データサンプルをバッチテンソルに照合
def string_collate(hparams):
    tokenizer = hparams.tokenizer
    pad_idx = tokenizer.pad_token_id

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_string, tgt_string in batch:
            inputs = hparams.tokenizer.batch_encode_plus(
                [src_string], max_length=hparams.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
            outputs = hparams.tokenizer.batch_encode_plus(
                [tgt_string], max_length=hparams.target_max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
            src_batch.append(inputs["input_ids"].squeeze())
            tgt_batch.append(outputs["input_ids"].squeeze())

        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    return collate_fn


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
    model_name_or_path='google/mt5-small',
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

    vocab_size = hparams.tokenizer.vocab_size
    PAD_IDX = hparams.tokenizer.pad_token_id
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

    NUM_EPOCHS = 2

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
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    torch.save(model.state_dict(), 'transformer_model.pt')
    print(generate(model, hparams.tokenizer, 'encourage: 難しいです'))
    # model.load_state_dict(torch.load('transformer_model.pt'))


# greedy search を使って翻訳結果 (シーケンス) を生成


def _greedy_decode(model, src, src_mask, max_len, beamsize, start_symbol, end_idx):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        # prob.size() の実行結果 : torch.Size([1, 1088]) => 1088 はTGT のVOCAV_SIZE
        prob = model.generator(out[:, -1])
        next_prob, next_word = prob.topk(k=beamsize, dim=1)

        next_word = next_word[:, 0]     # greedy なので、もっとも確率が高いものを選ぶ
        next_word = next_word.item()   # 要素の値を取得 (int に変換)

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_idx:
            break
    print(ys)
    return ys

# 翻訳


def generate(model, tokenizer, src_sentence: str):
    model.eval()
    inputs = tokenizer(src_sentence, max_length=128, truncation=True,
                       return_tensors='pt')   # input のtensor
    src = inputs['input_ids'].view(-1, 1)
    start_symbol = tokenizer.pad_token_id  # int(src[0])  # SOS_IDXの代わり
    end_idx = tokenizer.eos_token_id
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = _greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, beamsize=5, start_symbol=start_symbol, end_idx=end_idx)
    return tokenizer.decode(tgt_tokens.flatten())


if __name__ == '__main__':
    _main()

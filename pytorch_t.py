
import logging
import hashlib
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
#import sentencepiece as spm
import math

from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0

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
        # print('@vocab', src_vocab_size, tgt_vocab_size)
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


# greedy search を使って翻訳結果 (シーケンス) を生成


def _greedy_decode(model, src, src_mask, device, max_len, beamsize, start_symbol, end_idx):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
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
    return ys

# 翻訳


def _generate(model, tokenizer, device, bos_token_id: int, src_sentence: str):
    inputs = tokenizer(src_sentence, max_length=128, truncation=True,
                       return_tensors='pt')   # input のtensor
    src = inputs['input_ids'].view(-1, 1)
    end_idx = tokenizer.eos_token_id
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = _greedy_decode(
        model, src, src_mask, device,
        max_len=num_tokens + 5, beamsize=5,
        start_symbol=bos_token_id, end_idx=end_idx)
    return tokenizer.decode(tgt_tokens.flatten())


def md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def save_model(hparams, model, file='transformer-model.pt'):
    if len(hparams.bos_token) > 0:
        bos_token_id = int(hparams.tokenizer.encode(hparams.bos_token)[0])
    else:
        bos_token_id = 0
    torch.save(dict(
        tokenizer=hparams.tokenizer_name_or_path,
        additional_tokens=hparams.additional_tokens,
        bos_token_id=bos_token_id,
        num_encoder_layers=hparams.num_encoder_layers,
        num_decoder_layers=hparams.num_decoder_layers,
        emb_size=hparams.emb_size,
        nhead=hparams.nhead,
        vocab_size=hparams.vocab_size,
        fnn_hid_dim=hparams.fnn_hid_dim,
        model=model.state_dict(),
    ), file)
    logging.info(f'md5: {file} {md5(file)}')


def load_pretrained(filepath, device):
    logging.info(f'md5: {filepath} {md5(filepath)}')
    checkpoint = torch.load(filepath, map_location=device)
    print(list(checkpoint.keys()))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer'])
    tokenizer.add_tokens(checkpoint['additional_tokens'])
    # for tok in checkpoint['additional_tokens']:
    #     print(tok, tokenizer.vocab[tok], checkpoint['vocab_size'])
    model = Seq2SeqTransformer(
        checkpoint['num_encoder_layers'],
        checkpoint['num_decoder_layers'],
        checkpoint['emb_size'],
        checkpoint['nhead'],
        checkpoint['vocab_size'],
        checkpoint['vocab_size'],
        checkpoint['fnn_hid_dim']
    )
    model.load_state_dict(checkpoint['model'])
    model.train()
    return model


def load_nmt(filename='transformer-model.pt', device='cpu'):
    logging.info(f'md5: {filename} {md5(filename)}')
    device = torch.device(device)
    checkpoint = torch.load(filename, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer'])
    tokenizer.add_tokens(checkpoint['additional_tokens'])
    # for tok in checkpoint['additional_tokens']:
    #     print(tok, tokenizer.vocab[tok], checkpoint['vocab_size'])
    model = Seq2SeqTransformer(
        checkpoint['num_encoder_layers'],
        checkpoint['num_decoder_layers'],
        checkpoint['emb_size'],
        checkpoint['nhead'],
        checkpoint['vocab_size'],
        checkpoint['vocab_size'],
        checkpoint['fnn_hid_dim']
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    bos_token_id = checkpoint['bos_token_id']

    def generate_greedy(s: str) -> str:
        return _generate(model, tokenizer, device, bos_token_id, s)
    return generate_greedy

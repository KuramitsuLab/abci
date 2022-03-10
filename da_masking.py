import json
import builtins
import random
import keyword
import sys
from tokenize import tokenize, TokenError, open, generate_tokens, COMMENT, ENCODING, NL
from io import BytesIO

from pytokenizer import pycodify, tokenize_pycode


def denoising_objective(tokens, sep='', ratio=0.5):
    src = []
    tgt = []
    prev = None
    c = 0
    for token in tokens:
        if random.random() < ratio:
            if prev is not src:
                src.append(f'<extra_id_{c}>')
                prev = src
                c += 1
            tgt.append(token)
        else:
            src.append(token)
            if prev is not tgt:
                tgt.append(f'<extra_id_{c}>')
                prev = tgt
                c += 1
    return pycodify(src), pycodify(tgt)


def bert_style_masking(tokens, sep='', ratio=0.5):
    src = []
    tgt = []
    prev = None
    c = 0
    for token in tokens:
        if random.random() < ratio:
            if prev is not src:
                src.append(f'<extra_id_{c}>')
                prev = src
                c += 1
            tgt.append(token)
        else:
            src.append(token)
            tgt.append(token)
            prev = tgt
    return pycodify(src), pycodify(tgt)


def mask(s, tokenize=tokenize_pycode, sep='', ratio=0.5, masking=bert_style_masking):
    ss = tokenize(s)
    return masking(ss, sep=sep, ratio=ratio)


def transform_denoising_python(s, hparams):
    ss = tokenize_pycode(s)
    return denoising_objective(ss, ratio=hparams.masking_ratio)


def transform_bert_masking_python(s, hparams):
    ss = tokenize_pycode(s)
    return bert_style_masking(ss, ratio=hparams.masking_ratio)


def get_transform_masking(hparams):
    if hasattr(hparams, 'masking_style'):
        style = hparams.masking_style
        if style.lower() == 'bert':
            return transform_bert_masking_python
    return transform_denoising_python

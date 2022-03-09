import json
import builtins
import random
import keyword
import sys
from tokenize import tokenize, TokenError, open, generate_tokens, COMMENT, ENCODING, NL
from io import BytesIO


def codify(ss, sep=''):
    return sep.join(ss).strip().replace('  ', ' ')


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
    return codify(src), codify(tgt)


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
    return codify(src), codify(tgt)


VALUE = set(['True', 'False', 'None'])
SPACE = set(['return', ''])


def iskeyword(s):
    if s in VALUE:
        return False
    return keyword.iskeyword(s)


def append_token(ss, s):
    s = s.strip()
    if s in SPACE:
        return
    if iskeyword(s):
        ss.append(f' {s} ')
    elif len(s) > 0 and '\n' not in s:
        ss.append(s)


def tokenize_pycode(s):
    ss = []
    g = tokenize(BytesIO(s.encode('utf-8')).readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        #print(toknum, tokval)
        if toknum in (COMMENT, ENCODING):
            continue
        append_token(ss, tokval)
    return ss


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


def output(ss, sep=''):
    if len(ss) < 2 or (len(ss) == 2 and ss[-1] == ':'):
        #print('FIXME', ss)
        return
    code = codify(ss)
    # if code.startswith('pass'):
    #     print('FIXME', ss)
    if len(code) > 4 and not code.startswith('"""') and not code.startswith("'''"):
        try:
            tokenize_pycode(code)
            print(code)
        except TokenError:
            pass


def split_output(tokens, sep=''):
    prev = ''
    ss = []
    for toknum, tokval, _, _, _ in tokens:
        if toknum in (COMMENT, ENCODING):
            continue
        if toknum == NL or toknum == 4:
            if prev != ',' and prev != '(':
                output(ss, sep=sep)
                ss = []
            continue
        prev = tokval
        append_token(ss, tokval)
        #print(toknum, tokval)
    output(ss, sep=sep)


def tokenize_pyfile(file, sep=''):
    with open(file) as f:
        tokens = generate_tokens(f.readline)
        split_output(tokens, sep=sep)


def tokenize_text(file, sep=''):
    with builtins.open(file) as f:
        for line in f.readlines():
            line = line.strip()
            try:
                ss = tokenize_pycode(line.strip())
                output(ss, sep=sep)
            except TokenError:
                pass


def tokenize_jsonl(file, key='snippet', sep=''):
    with builtins.open(file) as f:
        for line in f.readlines():
            line = line.strip()
            data = json.loads(line)
            s = data[key]
            try:
                tokens = tokenize(BytesIO(s.encode('utf-8')).readline)
                split_output(tokens, sep=sep)
            except TokenError:
                pass
            except IndentationError:
                pass


def tokenize_ipynb(file, sep=''):
    with builtins.open(file) as f:
        data = json.loads(f.read())
        for d in data['cells']:
            if d.get('cell_type', '') != 'code':
                continue
            if 'source' in d:
                s = '\n'.join(d['source'])
                try:
                    if '%%' not in s:
                        tokens = tokenize(BytesIO(s.encode('utf-8')).readline)
                        split_output(tokens, sep=sep)
                except TokenError:
                    pass
                except IndentationError:
                    pass


if __name__ == '__main__':
    for file in sys.argv[1:]:
        if file.endswith('.py'):
            tokenize_pyfile(file)
        elif file.endswith('.jsonl'):
            # tokenize_jsonl(file, key='snippet')  # CoNaLa
            tokenize_jsonl(file, key='original_string')  # CodeSearchNet
        elif file.endswith('.ipynb'):
            tokenize_ipynb(file)  # CodeSearchNet
        else:
            tokenize_text(file)

import random
import keyword
import sys
from tokenize import tokenize, open, generate_tokens, COMMENT, ENCODING, NL
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


def tokenize_pycode(s):
    ss = []
    g = tokenize(BytesIO(s.encode('utf-8')).readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        #print(toknum, tokval)
        if toknum in (COMMENT, ENCODING):
            continue
        if keyword.iskeyword(tokval):
            ss.append(f' {tokval} ')
        else:
            ss.append(tokval)
    return ss


def mask(s, tokenize=tokenize_pycode, sep='', ratio=0.5, masking=bert_style_masking):
    ss = tokenize(s)
    return masking(ss, sep=sep, ratio=ratio)


mask('print(a, x in a) #hello', ratio=0.3)


def output(ss, sep=''):
    if len(ss) < 2:
        return
    code = sep.join(ss).strip().replace('  ', ' ')
    if len(code) > 0 and not code.startswith('"""') and not code.startswith("'''"):
        print(code)


def tokenize_pyfile(file, sep=''):
    with open(file) as f:
        tokens = generate_tokens(f.readline)
        prev = ''
        ss = []
        for toknum, tokval, _, _, _ in tokens:
            if toknum == COMMENT:
                continue
            if toknum == NL or toknum == 4:
                if prev != ',' and prev != '(':
                    output(ss, sep=sep)
                    ss = []
                continue
            prev = tokval
            if keyword.iskeyword(tokval):
                ss.append(f' {tokval} ')
            elif '\n' not in tokval:
                ss.append(tokval)
            #print(toknum, tokval)
        output(ss, sep=sep)


if __name__ == '__main__':
    for file in sys.argv[1:]:
        tokenize_pyfile(file)

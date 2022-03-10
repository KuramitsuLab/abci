import json
import builtins
import keyword
import sys
from tokenize import tokenize, TokenError, open, generate_tokens, COMMENT, ENCODING, NL
from io import BytesIO


def pycodify(ss, sep=''):
    return sep.join(ss).strip().replace('  ', ' ')


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


def pynormalize(s):
    return pycodify(tokenize_pycode(s))


def _outout(ss, sep=''):
    if len(ss) < 2 or (len(ss) == 2 and ss[-1] == ':'):
        return
    code = pycodify(ss)
    if len(code) > 4 and not code.startswith('"""') and not code.startswith("'''"):
        try:
            tokenize_pycode(code)
            print(code)
        except TokenError:
            pass


def _split_output(tokens, sep=''):
    prev = ''
    ss = []
    for toknum, tokval, _, _, _ in tokens:
        if toknum in (COMMENT, ENCODING):
            continue
        if toknum == NL or toknum == 4:
            if prev != ',' and prev != '(':
                _outout(ss, sep=sep)
                ss = []
            continue
        prev = tokval
        append_token(ss, tokval)
        #print(toknum, tokval)
    _outout(ss, sep=sep)


def _tokenize_pyfile(file, sep=''):
    with open(file) as f:
        tokens = generate_tokens(f.readline)
        _split_output(tokens, sep=sep)


def _tokenize_text(file, sep=''):
    with builtins.open(file) as f:
        for line in f.readlines():
            line = line.strip()
            try:
                ss = tokenize_pycode(line.strip())
                _outout(ss, sep=sep)
            except TokenError:
                pass


def _tokenize_jsonl(file, key='snippet', sep=''):
    with builtins.open(file) as f:
        for line in f.readlines():
            line = line.strip()
            data = json.loads(line)
            s = data[key]
            try:
                tokens = tokenize(BytesIO(s.encode('utf-8')).readline)
                _split_output(tokens, sep=sep)
            except TokenError:
                pass
            except IndentationError:
                pass


def _tokenize_ipynb(file, sep=''):
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
                        _split_output(tokens, sep=sep)
                except TokenError:
                    pass
                except IndentationError:
                    pass


if __name__ == '__main__':
    for file in sys.argv[1:]:
        if file.endswith('.py'):
            _tokenize_pyfile(file)
        elif file.endswith('.jsonl'):
            # tokenize_jsonl(file, key='snippet')  # CoNaLa
            _tokenize_jsonl(file, key='original_string')  # CodeSearchNet
        elif file.endswith('.ipynb'):
            _tokenize_ipynb(file)  # CodeSearchNet
        else:
            _tokenize_text(file)

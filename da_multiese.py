import re
import random
import pegtree as pg

GRAMMAR = '''

Sentense = { (Chunk/ Punc)* #Chunk }

ChunkGroup = {
    Chunk* #Chunk
}

Chunk = 
  / Order
  / Choice
  / Text

Order = {
  "{"
  ChunkGroup ([|/] ChunkGroup)*
  "}"
  #Order
}

Choice = {
  "["
  ChunkGroup ("|" ChunkGroup )*
  "]"
  #Choice
}

Text = {
  (!PUNC .)+
  #Text
} / { 
  &']'
  #Text
}

Punc = { PUNC #Text }
PUNC = [{}\[\]|/]

'''

parse_as_tree = pg.generate(pg.grammar(GRAMMAR))


class Chunk(object):
    chunks: tuple

    def __init__(self, chunks):
        self.chunks = chunks

    def __repr__(self):
        return ' '.join(repr(x) for x in self.chunks)

    def emit(self, **kw):
        return ''.join(c.emit(**kw) for c in self.chunks)


class Text(object):
    text: str

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return repr(self.text)

    def emit(self, **kw):
        return self.text


class Choice(object):
    chunks: tuple

    def __init__(self, chunks, pos):
        self.chunks = chunks
        self.pos = pos

    def __repr__(self):
        return '['+('|'.join(repr(x) for x in self.chunks))+']'

    def emit(self, **kw):
        choice = kw.get('choice', 0.5)
        r = random.random()
        if r < choice:
            size = len(self.chunks)
            index = int(size * (r / choice)) % size
            #print(index, repr(self.chunks[index]))
            return self.chunks[index].emit(**kw)
        return self.chunks[0].emit(**kw)


class Order(object):
    chunks: tuple

    def __init__(self, chunks, pos):
        self.chunks = chunks
        self.pos = pos

    def __repr__(self):
        return '{'+(' | '.join(repr(x) for x in self.chunks))+'}'

    def emit(self, **kw):
        shuffle = kw.get('shuffle', 0.3)
        r = random.random()
        if r < shuffle:
            chunks = list(self.chunks)
            random.shuffle(chunks)
        else:
            chunks = self.chunks
        return ''.join(c.emit(**kw) for c in chunks)


class ParsedResult(object):
    def __init__(self):
        self.c = 0
        self.result = None

    def count(self):
        self.c += 1
        return self.c-1

    def __repr__(self):
        return repr(self.result)

    def emit(self, **kw):
        return self.result.emit(**kw)


def _traverse_tree(tree, c):
    if tree == 'Text':
        return Text(str(tree))
    ss = []
    for t in tree:
        ss.append(_traverse_tree(t, c))
    if len(ss) > 0:
        if tree == 'Chunk':
            if len(ss) == 1:
                return ss[0]
            return Chunk(tuple(ss))
        if tree == 'Order':
            if len(ss) == 1:
                return ss[0]
            return Order(tuple(ss), c.count())
        if tree == 'Choice':
            if len(ss) == 1:
                ss.append(Text(''))
            return Choice(tuple(ss), c.count())
    return Text('')


def multiese_da(s, choice=0.5, shuffle=0.5):
    if '|' in s:
        tree = parse_as_tree(s)
        c = ParsedResult()
        c.result = _traverse_tree(tree, c)
        #print(repr(c.result), c.c, c.emit(choice=0.9))
        return c.emit(shuffle=shuffle, choice=choice)
    return s


BEGIN = '([^A-Za-z0-9]|^)'
END = ('(?![A-Za-z0-9]|$)')
VARPAT = re.compile(BEGIN+r'([a-z]+\d?)'+END)


def _replace(doc, oldnews):
    doc = re.sub(VARPAT, r'\1@\2@', doc)  # @s@
    for old, new in oldnews:
        doc = doc.replace(old, new)
    return doc.replace('@', '')


def _reverse_replace(doc, oldnews):
    for new, old in oldnews:
        doc = doc.replace(old, new)
    return doc.replace('</s>', '')


def transform_translate(pair, hparams):
    text, code = pair
    text = multiese_da(text,
                       choice=hparams.da_choice,
                       shuffle=hparams.da_shuffle)
    if random.ranom() < hparams.masking_ratio:
        names = [x[1] for x in VARPAT.findall(text+' ') if x[1] in code]
        d = {}
        oldnews = []
        for name in names:
            if name not in d:
                d[name] = f'<e{len(d)}>'
                oldnews.append((f'@{name}@', d[name]))
        text = _replace(text, oldnews).strip()
        code = _replace(code+' ', oldnews).strip()
    return text, code


def transform_multiese(pair, hparams):
    src, tgt = pair
    src = multiese_da(src,
                      choice=hparams.da_choice,
                      shuffle=hparams.da_shuffle)
    tgt = multiese_da(tgt,
                      choice=hparams.da_choice,
                      shuffle=hparams.da_shuffle)
    return src, tgt


def transform_masking(pair, hparams):
    src, tgt = pair
    tokenizer = hparams.tokenizer
    inputs = tokenizer.encode(src, add_special_tokens=False)
    ratio = hparams.masking_ratio if hasattr(
        hparams, 'masking_ratio') else 0.3
    src = []
    prev = None
    c = 0
    for id in inputs:
        if random.random() < ratio:
            if prev is not src:
                src.append(f'<extra_id_{c}>')
                prev = src
                c += 1
        else:
            src.append(f'{tokenizer.decode([id])}')
            prev = None
    return ''.join(src), tgt


TRANSFORMS = {
    '*': transform_multiese,
    'translate': transform_translate,
    'trans': transform_translate,
    'mask': transform_masking,
}


def transform_multitask(pair, hparams):
    if pair[0] is pair[1]:
        src, tgt = transform_masking(pair, hparams)
    else:
        prefix, _, _ = pair[0].partition(':')
        transform_fn = TRANSFORMS.get(prefix, transform_multiese)
        src, tgt = transform_fn(pair, hparams)
    return src, hparams.bos_token+tgt


def compose_testing(gen_fn, trans_prefix='trans'):
    def testing(src, tgt):
        if src.startswith(trans_prefix):
            src2 = src + ' '
            names = [x[1] for x in VARPAT.findall(src2) if x[1] in tgt]
            d = {}
            oldnews = []
            for name in names:
                if name not in d:
                    d[name] = f'<e{len(d)}>'
                    oldnews.append((f'@{name}@', d[name]))
            src2 = _replace(src2, oldnews).strip()
            src2 = gen_fn(src2)
            gen = _reverse_replace(src2, oldnews).replace('@', '').strip()
            return src, gen, tgt
        else:
            return src, gen_fn(src), tgt
    return testing


def transform_testing(pair, hparams):
    if isinstance(pair, tuple):
        src = multiese_da(pair[0])
        tgt = multiese_da(pair[1])
        return src, tgt
    return pair, pair  # TODO: masking


if __name__ == '__main__':
    multiese_da(']a')
    multiese_da('subを探す')
    multiese_da('{Aと|[||B]subを}探す')
    multiese_da('[Aと|subを]探す')
    multiese_da('{sのstartからend[|まで]の間で|subを}探す')

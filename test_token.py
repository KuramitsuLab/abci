from transformers import AutoTokenizer
from io import BytesIO
from tokenize import tokenize, open
import re

pattern = re.compile(r'[\(, .\+\-\)]')


def tokenize_pycode(code):
    try:
        ss = []
        tokens = tokenize(BytesIO(code.encode('utf-8')).readline)
        for toknum, tokval, _, _, _ in tokens:
            if toknum != 62 and tokval != '' and tokval != 'utf-8':
                ss.append(tokval)
        return ss
    except:
        return pattern.split(code)


#tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
#tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")


def _main():
    with open('python_corpus/mask_CoNaLa.txt') as f:

        for line in f.readlines()[:10]:
            line = line.strip()
            tokens = tokenize_pycode(line)
            inputs = tokenizer(line)   # input のtensor 列
            input_ids = inputs['input_ids']
            print(len(tokens), line)
            decoded = tokenizer.decode(input_ids).removesuffix('</s>')
            print(len(input_ids)-1, decoded)


_main()

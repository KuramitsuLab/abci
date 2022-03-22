
import csv
import sys

from transformers import AutoTokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration

#from pytorch_t import load_nmt
from train_mt5 import load_nmt
from da_multiese import compose_testing
from multiese2_test import test_code
import random


def test(tuples, gen_fn, w=None, sample=100):
    testing_fn = compose_testing(gen_fn)
    count = 0
    for src, tgt, params in random.sample(tuples, sample):
        src, gen, tgt = testing_fn(src, tgt)
        tgt_tested = test_code(tgt, params)
        gen_tested = test_code(gen, params)
        check = (1 if tgt_tested == gen_tested else 0)
        count += check
        results = (check, src, gen, tgt, gen_tested, tgt_tested)
        print(check, gen, tgt)
        if check == 0:
            print('\n', gen_tested)
            print('\n', tgt_tested)
        if w is not None:
            w.writerow(results)
    print(count, sample, count/sample)


def load_tsv(file, tuples):
    sep = ',' if file.endswith('.csv') else '\t'
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if len(row) > 3:
                tuples.append((row[2], row[1], row[3]))
    return tuples


def load_t5_generate(model_path):
    from train_mt5 import load_nmt
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # if '/mt5' in model_path:
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    # else:
    #     model = T5ForConditionalGeneration.from_pretrained(model_path)
    return load_nmt(model, tokenizer)


def _main():
    tuples = []
    gen_fn = None
    for f in sys.argv[1:]:
        if f.endswith('.tsv') or f.endswith('.tsv'):
            load_tsv(f, tuples)
        elif f.endswith('pt'):
            gen_fn = load_nmt(f)
        else:
            gen_fn = load_t5_generate(f)

    with open('test_code.tsv', 'w') as w:
        w = csv.writer(w, delimiter='\t')
        test(tuples, gen_fn, w, sample=100)


if __name__ == '__main__':
    _main()
